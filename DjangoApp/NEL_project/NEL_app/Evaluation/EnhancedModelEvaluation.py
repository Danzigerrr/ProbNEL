import json
import multiprocessing
import time
import os
from datetime import datetime
from typing import List

from tqdm import tqdm
from DjangoApp.NEL_project.NEL_app.Evaluation.DatasetLoader import DatasetLoader
from DjangoApp.NEL_project.NEL_app.Evaluation.TestDataset import TestDataset
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.Models.Text import Text
from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.DBpediaSearch import DBpediaSearch
from concurrent.futures import ProcessPoolExecutor, as_completed

from DjangoApp.NEL_project.NEL_app.NED_utlis.NEDHandler import NEDHandler
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


class EnhancedModelEvaluation:
    def __init__(self, dataset_path, ned_handler: NEDHandler, output_dir="evaluation_results"):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.ned_knowledge_graph = "dbpedia"
        self.DBPediaSearch = DBpediaSearch()
        self.ned_handler = ned_handler
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
        for text in tqdm(dataset.texts, desc="NER: Performing NER using golden annotations"):
            text_with_pred = Text(text.content)
            for original_golden_entity in text.entities:
                new_entity = Entity(
                    entity_label=original_golden_entity.entity_label,
                    entity_type="Undefined NER type - no NER model used in this configuration",
                    start_position=original_golden_entity.start_position,
                    end_position=original_golden_entity.end_position,
                    best_candidate_uri="It will be set during NED and evaluated during evaluation process",
                    probabilities=[]
                )
                text_with_pred.entities.append(new_entity)
            texts_with_golden_annotations.append(text_with_pred)
        return texts_with_golden_annotations

    def perform_ned_on_texts(self, texts_with_pred: List[Text]) -> List[Text]:
        use_first_n = 1

        texts_with_ned = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.ned_handler.perform_ned, text): idx for idx, text in enumerate(texts_with_pred)}
            results = {}
            with tqdm(total=len(texts_with_pred), desc="NED: Processing texts") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    pbar.update(1)
        for idx in sorted(results.keys()):
            texts_with_ned.append(results[idx])
        return texts_with_ned

    def evaluate_linking(self, texts_with_predictions: List[Text], golden_dataset: TestDataset):
        TP = FP = FN = 0
        for predicted_text, golden_text in zip(texts_with_predictions, golden_dataset.texts):
            for pred_entity, golden_entity in zip(predicted_text.entities, golden_text.entities):
                if golden_entity.target_uri:
                    if pred_entity.best_candidate_uri == golden_entity.target_uri:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FN += 1

        print(f"TP:{TP}")
        print(f"FP:{FP}")
        print(f"FN:{FN}")

        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1_score

    def run_enhanced_evaluation(self):
        dataset_file = self.load_evaluation_dataset_file()
        if not dataset_file:
            return

        start_time = time.time()
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_dataset_content(dataset_file, self.dataset_path)
        dataset_loader.print_dataset_info(dataset)

        texts_with_golden_annotations = self.generate_list_of_texts_with_named_entities(dataset)
        texts_with_predictions = self.perform_ned_on_texts(texts_with_golden_annotations)

        accuracy, precision, recall, f1_score = self.evaluate_linking(texts_with_predictions, dataset)
        self.save_results([accuracy, precision, recall, f1_score], start_time, dataset_loader)

    def save_results(self, ned_results, start_time, dataset_loader):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/enhanced_evaluation_{timestamp}.json"

        output_data = {
            "name": "Enhanced system evaluation",
            "configuration": {
                "dataset_path": self.dataset_path,
                "dataset_total_texts": dataset_loader.total_texts,
                "dataset_total_mentions": dataset_loader.total_mentions,
                "ned_knowledge_graph": self.ned_knowledge_graph,
                "execution_time_seconds": round(time.time() - start_time, 2)
            },
            "ned_results": {
                "accuracy": ned_results[0],
                "precision": ned_results[1],
                "recall": ned_results[2],
                "f1_score": ned_results[3]
            }
        }

        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)

        print(f"Evaluation results saved to: {output_filename}")
        print(output_data["ned_results"])


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset_path = "./EvaluationDatasets/ace2004_short.json"

    ner_config_not_used = NERConfig("tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context")
    ned_handler = NEDHandler(ner_config_not_used, "dbpedia", "xgboost", False)

    print("Running enhanced model evaluation with golden annotations...")
    evaluation = EnhancedModelEvaluation(dataset_path, ned_handler)
    evaluation.run_enhanced_evaluation()
