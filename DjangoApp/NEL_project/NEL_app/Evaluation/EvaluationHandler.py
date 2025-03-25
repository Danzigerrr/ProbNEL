from .EvaluationResults import EvaluationResults
from ..Models.Text import Text
import threading
import time
from tqdm import tqdm

class EvaluationHandler:

    def __init__(self, ner_handler, ned_handler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        :param ner_handler: An instance of NERHandler for Named Entity Recognition.
        :param ned_handler: An instance of NEDHandler for Named Entity Disambiguation.
        """
        self.ner = ner_handler
        self.ned = ned_handler
        self.evaluation_results = EvaluationResults()  # Initialize a single EvaluationResults object
        self.num_threads = 8 # Set the number of threads

    def run_test_on_dataset(self, dataset):
        """
        Run NER and NED on the dataset in parallel and evaluate prediction accuracy.
        :param dataset: A dataset containing Text objects with ground truth entities.
        :return: An EvaluationResults object.
        """
        start_time = time.time()

        self.process_texts_in_parallel(dataset)

        self.evaluation_results.finalize_scores()  # calculate the final scores
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
        return self.evaluation_results

    def process_texts_in_parallel(self, dataset):
        texts = dataset.texts
        num_texts = len(texts)

        threads = []
        with tqdm(total=num_texts, desc="Evaluating dataset") as pbar:
            def process_text_wrapper(text_ground_truth):
                self.process_text(text_ground_truth)
                pbar.update(1)

            for text_ground_truth in texts:
                thread = threading.Thread(target=process_text_wrapper, args=(text_ground_truth,))
                threads.append(thread)
                thread.start()

                # Limit the number of active threads
                if len(threads) >= self.num_threads:
                    threads[0].join()
                    threads.pop(0)

            for thread in threads:
                thread.join()

    def process_text(self, text_ground_truth):
        """
        Process a single text, including NER, NED, and evaluation.
        """
        text_content = text_ground_truth.content
        text_to_analyse = Text(text_content)

        self.ner.perform_ner(text_to_analyse)
        self.ned.perform_ned(text_to_analyse)

        with threading.Lock(): # ensure thread safety
            self.evaluate_entity_linking_in_text(text_to_analyse, text_ground_truth)

    def evaluate_entity_linking_in_text(self,
                                        text_to_analyse: Text,
                                        ground_truth_text: Text):
        """
        Evaluate NED by comparing ground truth entities with predicted entities.
        """
        self.evaluation_results.calculate_scores(text_to_analyse.entities, ground_truth_text.entities)