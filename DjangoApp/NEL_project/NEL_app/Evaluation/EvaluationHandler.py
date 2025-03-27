import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .EvaluationResults import EvaluationResults
from .TestText import TestText
import time
from tqdm import tqdm
import os
import csv
import json

from .TestDataset import TestDataset


class EvaluationHandler:
    def __init__(self, ner_handler, ned_handler, max_workers=4):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        :param ner_handler: An instance of NERHandler for Named Entity Recognition.
        :param ned_handler: An instance of NEDHandler for Named Entity Disambiguation.
        :param max_workers: Maximum number of threads to use for parallel processing.
        """
        self.ner = ner_handler
        self.ned = ned_handler
        self.evaluation_results = EvaluationResults()
        self.dataset = None
        self.max_workers = max_workers
        self.lock = threading.Lock()  # Ensure thread safety when updating results

    def run_test_on_dataset(self, dataset: TestDataset):
        """
        Run NER and NED on the dataset in parallel and evaluate prediction accuracy.
        :param dataset: A dataset containing Text objects with ground truth entities.
        :return: An EvaluationResults object.
        """
        self.dataset = dataset
        start_time = time.time()

        self.process_texts_parallel(dataset)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

        # save logs
        all_logs = []
        for text in dataset.texts:
            all_logs += text.logs

        os.makedirs("Evaluation_Logs", exist_ok=True)  # Ensure directory exists
        self.save_logs_to_json(all_logs)
        self.save_logs_to_csv(all_logs)

        return self.evaluation_results

    def process_texts_parallel(self, dataset):
        texts = dataset.texts
        num_texts = len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_text, text): text for text in texts}

            with tqdm(total=num_texts, desc="Evaluating dataset") as pbar:
                for future in as_completed(futures):
                    future.result()  # Wait for completion
                    pbar.update(1)

    def process_text(self, text_ground_truth):
        """
        Process a single text, including NER, NED, and evaluation.
        """
        text_content = text_ground_truth.content
        text_to_analyse = TestText(text_content)

        self.ner.perform_ner(text_to_analyse)
        self.ned.perform_ned(text_to_analyse)

        self.evaluate_entity_linking_in_text(text_to_analyse, text_ground_truth)

    def evaluate_entity_linking_in_text(self, text_to_analyse: TestText, ground_truth_text: TestText):
        """
        Evaluate NED by comparing ground truth entities with predicted entities.
        """
        self.evaluation_results.calculate_scores(text_to_analyse, text_to_analyse.entities, ground_truth_text.entities)

    def save_logs_to_csv(self, logs):
        """Saves the logs to a CSV file."""
        with open(f"Evaluation_Logs/evaluation_logs_{self.dataset.name}.csv", mode='w', newline='', encoding='utf-16') as file:
            writer = csv.writer(file)
            writer.writerow(["entity_label", "start_position", "end_position", "correct_prediction", "candidates", "matching_candidate_index"])
            for log in logs:
                candidates_str = json.dumps(log["candidates"])
                writer.writerow([log["entity_label"], log["start_position"], log["end_position"], log["correct_prediction"], candidates_str, log.get("matching_candidate_index", None)])

    def save_logs_to_json(self, logs):
        """Saves the logs to a JSON file."""
        with open(f"Evaluation_Logs/evaluation_logs_{self.dataset.name}.json", 'w', encoding='utf-16') as file:
            json.dump(logs, file, indent=4)
