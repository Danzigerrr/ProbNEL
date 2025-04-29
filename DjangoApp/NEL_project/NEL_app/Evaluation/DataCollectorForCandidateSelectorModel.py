import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from datetime import datetime
from typing import List

from tqdm import tqdm

from DjangoApp.NEL_project.NEL_app.Evaluation.DatasetLoader import DatasetLoader
from DjangoApp.NEL_project.NEL_app.Evaluation.TestDataset import TestDataset
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.Models.Text import Text
from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.DBpediaSearch import DBpediaSearch


class DataCollectorForCandidateSelectorModel:
    def __init__(self, dataset_path, output_dir="candidate_selection_data"):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.ned_knowledge_graph = "dbpedia"
        self.DBPediaSearch = DBpediaSearch()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_evaluation_dataset_file(self):
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def generate_list_of_texts_with_named_entities(self, dataset: TestDataset):
        texts_with_golden_annotations = []
        for text in dataset.texts:
            text_with_pred = Text(text.content)
            for original_golden_entity in text.entities:
                new_entity = Entity(
                    entity_label=original_golden_entity.entity_label,
                    entity_type="Undefined NER type",
                    start_position=original_golden_entity.start_position,
                    end_position=original_golden_entity.end_position,
                    best_candidate_uri=original_golden_entity.target_uri,
                    probabilities=[]
                )
                text_with_pred.entities.append(new_entity)
            texts_with_golden_annotations.append(text_with_pred)
        return texts_with_golden_annotations

    def fetch_candidates(self, entity_label, top_n):
        return DBpediaSearch().search_by_entity_surface_form(entity_label, top_n)

    def collect_top_candidates_for_named_entities(self, texts_with_golden_annotations: List[Text], top_n_candidates=10) -> List[Text]:
        with ProcessPoolExecutor() as executor:
            futures = []
            for text in texts_with_golden_annotations:
                for entity in text.entities:
                    futures.append((entity, executor.submit(self.fetch_candidates, entity.entity_label, top_n_candidates)))

            for entity, future in tqdm(futures, desc="Collecting candidates"):
                entity.candidates = future.result()

        return texts_with_golden_annotations

    def run_entity_linking_data_collection(self):
        dataset_file = self.load_evaluation_dataset_file()
        if not dataset_file:
            return

        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_dataset_content(dataset_file, self.dataset_path)
        dataset_loader.print_dataset_info(dataset)

        texts_with_golden_annotations = self.generate_list_of_texts_with_named_entities(dataset)
        texts_with_predictions = self.collect_top_candidates_for_named_entities(texts_with_golden_annotations)

        self.save_results(dataset_loader, texts_with_golden_annotations, texts_with_predictions)

    def save_results(self, dataset_loader, golden_texts, predicted_texts):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/candidates_data_collector_{timestamp}.json"

        golden_annotations = [
            {
                "content": text.content,
                "entities": [
                    {
                        "entity_label": ent.entity_label,
                        "start_position": ent.start_position,
                        "end_position": ent.end_position,
                        "best_candidate_uri": ent.best_candidate_uri
                    }
                    for ent in text.entities
                ]
            }
            for text in golden_texts
        ]

        predictions = [
            {
                "content": text.content,
                "entities": [
                    {
                        "entity_label": ent.entity_label,
                        "start_position": ent.start_position,
                        "end_position": ent.end_position,
                        "candidates": [
                            {
                                "label": c.label,
                                "ontology_types": c.ontology_types,
                                "comment": c.comment,
                                "uri": c.uri,
                                "ref_count": c.ref_count
                            } for c in ent.candidates
                        ]
                    }
                    for ent in text.entities
                ]
            }
            for text in predicted_texts
        ]

        output_data = {
            "name": "Enhanced system evaluation",
            "configuration": {
                "dataset_path": self.dataset_path,
                "dataset_total_texts": dataset_loader.total_texts,
                "dataset_total_mentions": dataset_loader.total_mentions,
                "ned_knowledge_graph": self.ned_knowledge_graph,
            },
            "golden_annotations": golden_annotations,
            "predictions": predictions
        }

        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)

        print(f"Evaluation results saved to: {output_filename}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset_path = "./EvaluationDatasets/ace2004_full.json"

    print("Running data collector with golden annotations...")
    data_collector = DataCollectorForCandidateSelectorModel(dataset_path)
    data_collector.run_entity_linking_data_collection()
