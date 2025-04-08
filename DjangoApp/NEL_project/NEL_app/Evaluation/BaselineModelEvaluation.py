import json
import multiprocessing
import time
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from DjangoApp.NEL_project.NEL_app.Evaluation.DatasetLoader import DatasetLoader
from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.DBpediaSearch import DBpediaSearch
from concurrent.futures import ProcessPoolExecutor, as_completed

class BaselineModelEvaluation:
    def __init__(self, dataset_path, output_dir="evaluation_results"):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.ned_knowledge_graph = "dbpedia"
        self.DBPediaSearch = DBpediaSearch()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.correct_candidate_ranks = []  # Store positions of correct candidates

    def load_evaluation_dataset_file(self):
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def run_baseline_evaluation(self):
        dataset_file = self.load_evaluation_dataset_file()
        if not dataset_file:
            return

        start_time = time.time()
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
                result, rank = future.result()
                if result == 'TP':
                    TP += 1
                elif result == 'FP':
                    FP += 1
                elif result == 'FN':
                    FN += 1
                if rank is not None:
                    self.correct_candidate_ranks.append(rank)

        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ned_results = [accuracy, precision, recall, f1_score]

        self.save_results(ned_results, start_time, dataset_loader)
        self.plot_candidate_distribution()

    def process_entity(self, entity):
        entity_label = entity.entity_label
        candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label, 10)
        if candidates:
            uris = [c.uri for c in candidates]
            if entity.target_uri in uris:
                rank = uris.index(entity.target_uri) + 1  # 1-based index
                return 'TP', rank
            else:
                return 'FP', None
        else:
            return 'FN', None

    def save_results(self, ned_results, start_time, dataset_loader):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/evaluation_{timestamp}.json"

        output_data = {
            "name": "Baseline system evaluation",
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

    def plot_candidate_distribution(self):
        if not self.correct_candidate_ranks:
            print("No correct candidate ranks to plot.")
            return

        total_entities = len(self.correct_candidate_ranks)

        # Histogram Plot (Absolute Frequencies)
        plt.figure(figsize=(8, 6))
        plt.hist(self.correct_candidate_ranks, bins=range(1, 12), align='left', rwidth=0.8,
                 color='skyblue', edgecolor='black')
        plt.xlabel('Correct Candidate Rank Position')
        plt.ylabel('Frequency')
        plt.title('Distribution of Correct Candidate Positions')
        plt.xticks(range(1, 11))
        hist_filename = f"{self.output_dir}/distribution_{self.dataset_name}_top10.png"
        plt.savefig(hist_filename)
        plt.close()
        print(f"Candidate distribution plot saved to: {hist_filename}")

        # Cumulative Frequency Plot (as Percentages)
        values, counts = np.unique(self.correct_candidate_ranks, return_counts=True)
        sorted_ranks = np.arange(1, 11)
        freq_dict = dict(zip(values, counts))
        freq = [freq_dict.get(rank, 0) for rank in sorted_ranks]
        cumulative = np.cumsum(freq)
        cumulative_percentage = (cumulative / total_entities) * 100

        plt.figure(figsize=(8, 6))
        plt.plot(sorted_ranks, cumulative_percentage, marker='o', linestyle='-', color='green')
        plt.xlabel('Rank Position')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative % of Correct Candidates Found at Top-k Ranks')
        plt.xticks(range(1, 11))
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(True, linestyle='--', alpha=0.7)
        cumulative_filename = f"{self.output_dir}/cumulative_distribution_{self.dataset_name}_top10.png"
        plt.savefig(cumulative_filename)
        plt.close()
        print(f"Cumulative candidate percentage plot saved to: {cumulative_filename}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    dataset_path = "./EvaluationDatasets/aida_test_full.json"

    print("Running baseline candidate selection evaluation...")
    evaluation = BaselineModelEvaluation(dataset_path)
    evaluation.run_baseline_evaluation()
