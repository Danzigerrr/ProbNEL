import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .EvaluationResults import EvaluationResults
from .EvaluationLogs import EvaluationLogs
from .EvaluationNER import NEREvaluator
from ..Models.Text import Text
from .TestDataset import TestDataset


class EvaluationHandler:
    def __init__(self, ner_handler, ned_handler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        """
        self.ner = ner_handler
        self.ned = ned_handler
        self.evaluation_results = EvaluationResults()
        self.evaluation_logs = EvaluationLogs()
        self.ner_evaluation = NEREvaluator()
        self.dataset = None
        self.max_threads = 4  # Set maximum number of threads

    def run_test_on_dataset(self, dataset: TestDataset):
        """
        Run NER and NED on the dataset in parallel, then evaluate results sequentially.
        """
        self.dataset = dataset
        start_time = time.time()

        processed_texts = self.process_texts_parallel(dataset)

        # Evaluation should be done sequentially
        self.process_results_sequential(processed_texts)
        self.evaluation_results.finalize_scores()
        self.evaluation_logs.save_logs_to_files()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

        return self.evaluation_results

    def process_texts_parallel(self, dataset):
        """Processes texts in parallel, collecting results for sequential evaluation."""
        texts = dataset.texts
        results = []

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_text = {executor.submit(self.process_text, text): (idx, text) for idx, text in enumerate(texts)}

            with tqdm(total=len(texts), desc="Processing texts") as pbar:
                for future in as_completed(future_to_text):
                    idx, text_ground_truth = future_to_text[future]
                    text_to_analyse = future.result()
                    results.append((idx, text_to_analyse, text_ground_truth))
                    pbar.update(1)

        # Sort results by index to maintain original order before sequential processing
        return sorted(results, key=lambda x: x[0])

    def process_text(self, text_ground_truth):
        """
        Process a single text with NER and NED.
        """
        text_content = text_ground_truth.content
        text_with_pred = Text(text_content)

        text_with_pred = self.ner.perform_ner(text_with_pred)
        text_with_pred = self.ned.perform_ned(text_with_pred)

        return text_with_pred

    def process_results_sequential(self, processed_texts):
        """Evaluates entity linking results sequentially."""
        with tqdm(total=len(processed_texts), desc="Evaluating results") as pbar:
            for _, text_to_analyse, text_ground_truth in processed_texts:
                self.evaluate_entity_linking_in_text(text_to_analyse, text_ground_truth)
                pbar.update(1)

    def evaluate_entity_linking_in_text(self, text_to_analyse: Text, text_ground_truth: Text):
        """Evaluate NED by comparing ground truth entities with predicted entities."""
        self.evaluation_results.evaluate_ned_process(text_to_analyse.entities, text_ground_truth.entities)
        self.evaluation_logs.create_logs(text_to_analyse.entities, text_ground_truth.entities)
        self.ner_evaluation.evaluate_ner(text_to_analyse.entities, text_ground_truth.entities)
