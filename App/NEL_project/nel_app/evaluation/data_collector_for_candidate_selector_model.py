import json
import os

from datetime import datetime
from typing import List

from tqdm import tqdm
import multiprocessing
from pathlib import Path

from ..evaluation.dataset_loader import DatasetLoader
from ..models.test_dataset import TestDataset
from ..models.entity import Entity
from ..models.text import Text
from App.NEL_project.nel_app.ned_utlis.dbpedia.dbpedia_search import DBpediaSearch
from App.NEL_project.nel_app.ner_utils.ner_config import NERConfig
from App.NEL_project.nel_app.ner_utils.ner_handler import NERHandler

class DataCollectorForCandidateSelectorModel:
    def __init__(self, dataset_path: str, ner_model_name: Optional[str], output_dir="candidate_selection_data"):
        self.dataset_path = dataset_path
        # short dataset name
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.DBPediaSearch = DBpediaSearch()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Determine whether to use NER model or golden annotations
        self.use_golden_annotations = ner_model_name is None
        self.ner_model_name = ner_model_name
        # short ner name if exists
        self.ner_short = ner_model_name.split('/')[-1] if ner_model_name else 'golden'
        if not self.use_golden_annotations:
            self.ner_config = NERConfig(ner_model_name)
            self.ner_handler = NERHandler(self.ner_config)
            self.ner_accuracy = {"correct": 0, "total": 0}

    def load_evaluation_dataset_file(self):
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def perform_ner_on_texts(self, dataset: TestDataset) -> List[Text]:
        self.ner_handler.ner_config.tagger_model.try_cuda()
        texts_with_filtered = []
        for gold_text in tqdm(dataset.texts, desc="Performing NER and filtering predictions"):
            pred_text = self.ner_handler.perform_ner(Text(gold_text.content))
            ground_keys = {(e.entity_label, e.start_position, e.end_position): e for e in gold_text.entities}
            filtered = []
            for e in pred_text.entities:
                key = (e.entity_label, e.start_position, e.end_position)
                self.ner_accuracy["total"] += 1
                if key in ground_keys:
                    # correct detection
                    self.ner_accuracy["correct"] += 1
                    # assign best candidate from gold
                    e.best_candidate_uri = ground_keys[key].target_uri
                    filtered.append(e)
            pred_text.entities = filtered
            texts_with_filtered.append(pred_text)
        self.ner_handler.ner_config.tagger_model.cpu()
        return texts_with_filtered

    def generate_golden_texts(self, dataset: TestDataset) -> List[Text]:
        texts = []
        for text in dataset.texts:
            t = Text(text.content)
            for e in text.entities:
                ent = Entity(
                    entity_label=e.entity_label,
                    entity_type=e.entity_type,
                    start_position=e.start_position,
                    end_position=e.end_position,
                    best_candidate_uri=e.target_uri,
                    probabilities=[]
                )
                t.entities.append(ent)
            texts.append(t)
        return texts

    def collect_candidates(self, texts: List[Text], top_n=10) -> List[Text]:
        for t in tqdm(texts, desc="Fetching candidates"):
            for e in t.entities:
                e.candidates = self.DBPediaSearch.search_by_entity_surface_form(e.entity_label, top_n)
        return texts

    def save_results(self, dataset_loader, texts: List[Text]):
        # Consolidate into single list of named_entities
        records = []
        for t in texts:
            for e in t.entities:
                rec = {
                    "content": t.content,
                    "entity_label": e.entity_label,
                    "start_position": e.start_position,
                    "end_position": e.end_position,
                    "best_candidate_uri": e.best_candidate_uri,
                }
                if not self.use_golden_annotations:
                    rec["probabilities"] = e.probabilities
                # attach candidates
                rec["candidates"] = [
                    {
                        "label": c.label,
                        "ontology_types": c.ontology_types,
                        "comment": c.comment,
                        "uri": c.uri,
                        "ref_count": c.ref_count,
                        **({"dbpedia_score": getattr(c, 'dbpedia_score', None)} if hasattr(c, 'dbpedia_score') else {})
                    }
                    for c in e.candidates
                ]
                records.append(rec)

        output = {
            "name": "Data Collector for candidate selector model",
            "configuration": {
                "dataset_path": self.dataset_path,
                "use_golden_annotations": self.use_golden_annotations,
                "ner_model_name": self.ner_model_name,
            },
            "named_entities": records
        }
        # include dataset and ner short names in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{self.dataset_name}_{self.ner_short}_{timestamp}.json"
        out_path = os.path.join(self.output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Saved filtered results to {out_path}")

    def run_data_collection(self):
        data = self.load_evaluation_dataset_file()
        if not data:
            return
        loader = DatasetLoader()
        ds = loader.load_dataset_content(data, self.dataset_path)
        loader.print_dataset_info(ds)

        if self.use_golden_annotations:
            texts = self.generate_golden_texts(ds)
        else:
            # count total golden entities
            total_golden = sum(len(text.entities) for text in ds.texts)
            # perform NER and filter only correct ones
            texts = self.perform_ner_on_texts(ds)
            # calculate and display counts and accuracy
            correct = self.ner_accuracy["correct"]
            total_pred = self.ner_accuracy["total"]
            acc = correct / total_pred if total_pred > 0 else 0
            print(f"Total golden entities: {total_golden}")
            print(f"Correctly identified: {correct} of {total_pred} preds, accuracy {acc:.4f}")

        # fetch candidates for the filtered entities
        texts = self.collect_candidates(texts)
        self.save_results(loader, texts)
import gc
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    dataset_paths = [
        # "./evaluation_datasets/aida_train_converted.json",
        "./evaluation_datasets/aida_test_full.json",
        "./evaluation_datasets/ace2004_full.json"
    ]

    ner_models = [
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context",
        "tomaarsen/span-marker-roberta-large-ontonotes5",
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
    ]


    for dp in dataset_paths:
        for nm in ner_models:
            print(f"Running for {Path(dp).stem} with {nm}")
            dc = DataCollectorForCandidateSelectorModel(dp, nm)
            dc.run_data_collection()

            # flush large variables and force garbage collection
            del dc
            gc.collect()

