from .EvaluationResults import EvaluationResults
from ..classes import Text


class EvaluationHandler:
    def __init__(self, ner_handler, ned_handler):
        """
        Initialize the EvaluationHandler with handlers for NER and NED.
        :param ner_handler: An instance of NERHandler for Named Entity Recognition.
        :param ned_handler: An instance of NEDHandler for Named Entity Disambiguation.
        """
        self.ner_handler = ner_handler
        self.ned_handler = ned_handler

    def run_test_on_dataset(self, dataset):
        """
        Run NER and NED on the dataset and evaluate prediction accuracy.
        :param dataset: A dataset containing Text objects with ground truth entities.
        :return: An EvaluationResults object containing evaluation metrics.
        """
        evaluation_results = EvaluationResults()

        for ground_truth_text in dataset.texts:
            print(f"Processing text: {ground_truth_text.content[:50]}")

            # Perform Named Entity Recognition (NER)
            sentence_with_ner = self.ner_handler.process_text(ground_truth_text.content)

            # Create a new Text object to hold the predicted entities
            predicted_text = Text(ground_truth_text.content)

            # Extract entities from NER and add them to the Text object
            found_entities = self.ner_handler.extract_entities(sentence_with_ner, predicted_text)
            for entity in found_entities:
                predicted_text.add_entity(entity)

            # Perform Named Entity Disambiguation (NED)
            self.ned_handler.search_entities(sentence_with_ner, predicted_text)

            # Extract entities for comparison
            predicted_entities = self.get_entities_from_text(predicted_text)
            ground_truth_entities = self.get_entities_from_text(ground_truth_text)

            print(f"Predicted entities: {predicted_entities}")
            print(f"Ground truth entities: {ground_truth_entities}")

            # Evaluate NER and NED
            for ground_truth_entity in ground_truth_entities:
                matching_entity = next(
                    (
                        pred
                        for pred in predicted_entities
                        if pred["entity_label"] == ground_truth_entity["entity_label"]
                           and pred["start_position"] == ground_truth_entity["start_position"]
                           and pred["end_position"] == ground_truth_entity["end_position"]
                    ),
                    None,
                )

                evaluation_results.update_metrics(ground_truth_entity, matching_entity)

        # Finalize evaluation results
        evaluation_results.finalize()
        return evaluation_results

    def get_entities_from_text(self, text):
        """
        Extracts entities from a Text object.
        :param text: A Text object containing entities.
        :return: A list of dictionaries representing entities.
        """
        return [
            {
                "entity_label": e.entity_label,
                "start_position": e.start_position,
                "end_position": e.end_position,
                "uri": e.uri,
            }
            for e in text.entities
        ]
