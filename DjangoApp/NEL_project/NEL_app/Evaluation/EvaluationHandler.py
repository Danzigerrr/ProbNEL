import time
from tqdm import tqdm
from .EvaluationResults import EvaluationResults
from .EvaluationLogs import EvaluationLogs
from ..Models.Text import Text
from .TestDataset import TestDataset


class EvaluationHandler:
    def __init__(self, ner_handler, ned_handler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        :param ner_handler: An instance of NERHandler for Named Entity Recognition.
        :param ned_handler: An instance of NEDHandler for Named Entity Disambiguation.
        """
        self.ner = ner_handler
        self.ned = ned_handler
        self.evaluation_results = EvaluationResults()
        self.evaluation_logs = EvaluationLogs()
        self.dataset = None

    def run_test_on_dataset(self, dataset: TestDataset):
        """
        Run NER and NED on the dataset sequentially and evaluate prediction accuracy.
        :param dataset: A dataset containing Text objects with ground truth entities.
        :return: An EvaluationResults object.
        """
        self.dataset = dataset
        start_time = time.time()

        self.process_texts_sequential(dataset)

        self.evaluation_results.finalize_scores()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

        return self.evaluation_results

    def process_texts_sequential(self, dataset):
        """Processes texts sequentially, updating progress with tqdm."""
        texts = dataset.texts
        num_texts = len(texts)

        with tqdm(total=num_texts, desc="Evaluating dataset") as pbar:
            for text in texts:
                self.process_text(text)
                pbar.update(1)

    def process_text(self, text_ground_truth):
        """
        Process a single text, including NER, NED, and evaluation.
        """
        text_content = text_ground_truth.content
        text_to_analyse = Text(text_content)

        self.ner.perform_ner(text_to_analyse)
        self.ned.perform_ned(text_to_analyse)

        self.evaluate_entity_linking_in_text(text_to_analyse, text_ground_truth)
        self.evaluation_logs.create_logs(text_to_analyse.entities, text_ground_truth.entities)
        self.evaluation_logs.save_logs_to_files()

    def evaluate_entity_linking_in_text(self, text_to_analyse: Text, text_ground_truth: Text):
        """
        Evaluate NED by comparing ground truth entities with predicted entities.
        """
        self.evaluation_results.calculate_scores(text_to_analyse, text_to_analyse.entities, text_ground_truth.entities)


