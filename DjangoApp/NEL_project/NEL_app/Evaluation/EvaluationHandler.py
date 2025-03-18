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
        :return: A tuple containing NEREvaluationResults and EvaluationResults objects.
        """
        ner_evaluation_results = EvaluationResults()
        ned_evaluation_results = EvaluationResults()

        for ground_truth_text in dataset.texts:
            text_content = ground_truth_text.content
            print(f"Processing text: {text_content[:50]}...")

            text_obj = Text(text_content)

            # perform ner
            self.ner.perform_ner(text_obj)

            # evaluate ner
            self.evaluate_ner(ner_evaluation_results, text_obj, ground_truth_text)

            # perform ned
            self.ned.perform_ned(text_obj)

            # evaluate ned
            self.evaluate_ned(ned_evaluation_results, text_obj, ground_truth_text)

        # Finalize evaluation results
        ner_evaluation_results.calculate_accuracy()
        ned_evaluation_results.finalize()

        return ner_evaluation_results, ned_evaluation_results

    def evaluate_ner(self, ner_evaluation_results: EvaluationResults, text_obj: Text, ground_truth_text: Text):
        # Evaluate NER
        for ground_truth_entity in ground_truth_text.entities:
            correct_prediction = False
            for pred in text_obj.entities:
                if (pred.entity_label == ground_truth_entity.entity_label
                        and pred.start_position == ground_truth_entity.start_position
                        and pred.end_position == ground_truth_entity.end_position
                        and pred.entity_type == ground_truth_entity.entity_type):
                    correct_prediction = True

            ner_evaluation_results.update_metrics(correct_prediction)

    def evaluate_ned(self, ned_evaluation_results: EvaluationResults, text_obj: Text, ground_truth_text: Text):
        """
        Evaluate NED by comparing ground truth entities with predicted entities.
        """
        for ground_truth_entity in ground_truth_text.entities:
            correct_prediction = False
            for pred in text_obj.entities:
                if (pred.entity_label == ground_truth_entity.entity_label
                        and pred.start_position == ground_truth_entity.start_position
                        and pred.end_position == ground_truth_entity.end_position
                        and pred.best_candidate_uri == ground_truth_entity.best_candidate_uri):
                    correct_prediction = True

            ned_evaluation_results.update_metrics(correct_prediction)
