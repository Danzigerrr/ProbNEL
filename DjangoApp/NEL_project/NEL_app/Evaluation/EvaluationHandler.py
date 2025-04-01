import time
from typing import List

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .EvaluationNED import EvaluationNED
from .EvaluationLogs import EvaluationLogs
from .EvaluationNER import EvaluationNER
from ..Models.Text import Text
from .TestDataset import TestDataset
from ..NED_utlis.NEDHandler import NEDHandler
from ..NER_utils.NERHandler import NERHandler


class EvaluationHandler:
    def __init__(self, ner_handler: NERHandler, ned_handler: NEDHandler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        """
        self.ner = ner_handler
        self.ned = ned_handler
        self.ned_evaluation = EvaluationNED()
        self.ner_evaluation = EvaluationNER()
        self.evaluation_logs = EvaluationLogs()
        self.dataset = None
        self.max_threads = 16

    def run_test_on_dataset(self, dataset: TestDataset):
        """
        Run NER and NED on the dataset in parallel, then evaluate results sequentially.
        """
        self.dataset = dataset
        start_time = time.time()

        print("STATUS - start NER\n")
        texts_with_pred = self.perform_ner_on_texts(dataset)
        print("STATUS - finished NER\n")

        print("STATUS - start NED\n")
        texts_with_pred = self.perform_ned_on_texts(texts_with_pred)
        print("STATUS - finished NED\n")

        # Evaluation should be done sequentially
        self.evaluate_predictions(texts_with_pred, dataset)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

        return self.ned_evaluation, self.ner_evaluation

    def perform_ner_on_texts(self, dataset: TestDataset):
        """
        Process a single text with NER and NED.
        """
        self.ner.ner_config.tagger_model.try_cuda()

        texts_with_pred = []

        for text in dataset.texts:
            text_content = text.content
            text_with_pred = self.ner.perform_ner(Text(text_content))
            texts_with_pred.append(text_with_pred)

        self.ner.ner_config.tagger_model.cpu()

        return texts_with_pred

    def perform_ned_on_texts(self, texts_with_pred: List[Text]) -> List[Text]:
        """
        Process multiple texts with NED in parallel.
        """
        texts_with_ned = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all tasks concurrently.
            futures = {executor.submit(self.ned.perform_ned, text): idx for idx, text in enumerate(texts_with_pred)}
            # If order is important, collect results in a dict.
            results = {}
            with tqdm(total=len(texts_with_pred), desc="Processing texts") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    pbar.update(1)

        # Optionally, sort the results based on original order.
        for idx in sorted(results.keys()):
            texts_with_ned.append(results[idx])

        return texts_with_ned


    def evaluate_predictions(self, processed_texts: List[Text], dataset: TestDataset):
        """Evaluates entity linking results sequentially."""
        with tqdm(total=len(processed_texts), desc="Evaluating results") as pbar:
            for text_to_analyse, text_ground_truth in zip(processed_texts, dataset.texts):
                self.evaluate_entity_linking_in_text(text_to_analyse, text_ground_truth)
                pbar.update(1)

        self.ned_evaluation.finalize_scores()
        self.evaluation_logs.save_logs_to_files()

    def evaluate_entity_linking_in_text(self, text_to_analyse: Text, text_ground_truth: Text):
        """Evaluate NED by comparing ground truth entities with predicted entities."""
        self.ned_evaluation.evaluate_ned_process(text_to_analyse.entities, text_ground_truth.entities)
        self.evaluation_logs.create_logs(text_to_analyse.entities, text_ground_truth.entities)
        self.ner_evaluation.evaluate_ner(text_to_analyse.entities, text_ground_truth.entities)
