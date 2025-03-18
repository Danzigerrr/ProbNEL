from .EvaluationResults import EvaluationResults
from ..Models.Text import Text


class EvaluationHandler:

    def __init__(self, ner_handler, ned_handler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        :param ner_handler: An instance of NERHandler for Named Entity Recognition.
        :param ned_handler: An instance of NEDHandler for Named Entity Disambiguation.
        """
        self.ner = ner_handler
        self.ned = ned_handler

    def run_test_on_dataset(self, dataset):
        """
        Run NER and NED on the dataset and evaluate prediction accuracy.
        :param dataset: A dataset containing Text objects with ground truth entities.
        :return: A list of EvaluationResults objects.
        """
        all_evaluation_results = []
        for text_ground_truth in dataset.texts:
            text_content = text_ground_truth.content
            print(f"Processing text: {text_content[:50]}...")

            text_to_analyse = Text(text_content)

            self.ner.perform_ner(text_to_analyse)
            self.ned.perform_ned(text_to_analyse)

            evaluation_results = evaluate_entity_linking(text_to_analyse, text_ground_truth)
            all_evaluation_results.append(evaluation_results)

        return all_evaluation_results


def evaluate_entity_linking(text_to_analyse: Text,
                            ground_truth_text: Text):
    """
    Evaluate NED by comparing ground truth entities with predicted entities.
    """
    evaluation_results = EvaluationResults()
    evaluation_results.calculate_scores(text_to_analyse.entities, ground_truth_text.entities)

    return evaluation_results


