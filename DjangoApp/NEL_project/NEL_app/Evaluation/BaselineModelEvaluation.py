import json
import multiprocessing
import time
import os
from datetime import datetime
from tqdm import tqdm
from DjangoApp.NEL_project.NEL_app.Evaluation.DatasetLoader import DatasetLoader
from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.DBpediaSearch import DBpediaSearch
from concurrent.futures import ProcessPoolExecutor, as_completed

class BaselineModelEvaluation:
    def __init__(self, dataset_path, output_dir="evaluation_results"):
        self.dataset_path = dataset_path
        self.ned_knowledge_graph = "dbpedia"
        self.DBPediaSearch = DBpediaSearch()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    def load_evaluation_dataset_file(self):
        """Load the dataset from a JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None


    def run_baseline_evaluation(self):
        """Run evaluation on the dataset."""
        dataset_file = self.load_evaluation_dataset_file()
        if not dataset_file:
            return

        start_time = time.time()
        # Load dataset using DatasetLoader
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_dataset_content(dataset_file, self.dataset_path)
        dataset_loader.print_dataset_info(dataset)

        TN = TP = FP = FN = 0

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for text in tqdm(dataset.texts, desc="Processing Texts"):
                for entity in tqdm(text.entities, desc="Processing Entities", leave=False):
                    futures.append(executor.submit(self.process_entity, entity))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Entities"):
                result = future.result()
                if result == 'TP':
                    TP += 1
                elif result == 'FP':
                    FP += 1
                elif result == 'FN':
                    FN += 1

        # Calculate metrics
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ned_results = [accuracy, precision, recall, f1_score]
        # Save results
        self.save_results(ned_results, start_time, dataset_loader)

    def process_entity(self, entity):
        entity_label = entity.entity_label
        candidate = self.DBPediaSearch.search_by_entity_surface_form(entity_label, 1)
        if candidate:
            predicted_uri = candidate[0].uri
            if entity.target_uri == predicted_uri:
                return 'TP'
            else:
                return 'FP'
        else:
            return 'FN'


    def save_results(self, ned_results, start_time, dataset_loader):
        """Save evaluation results to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/evaluation_{timestamp}.json"

        output_data = {
            "name:": "Baseline system evaluation",
            "configuration": {
                "dataset_path": self.dataset_path,
                "dataset_total_texts": dataset_loader.total_texts,
                "dataset_total_mentions": dataset_loader.total_mentions,
                "ned_knowledge_graph": self.ned_knowledge_graph,
                "execution_time_seconds": round(time.time() - start_time, 2)
            },
            "ned_results":{
                "accuracy": ned_results[0],
                "precision": ned_results[1],
                "recall": ned_results[2],
                "f1_score": ned_results[3]
            }
        }

        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)

        print(f"Evaluation results saved to: {output_filename}:")
        print(output_data["ned_results"])

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset_path = "./EvaluationDatasets/aida_test_full.json"

    print(f"Running baseline candidate selection evaluation with:")

    evaluation = BaselineModelEvaluation(dataset_path)

    evaluation.run_baseline_evaluation()

