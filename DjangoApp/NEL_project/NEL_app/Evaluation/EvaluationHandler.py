from .EvaluationResults import EvaluationResults
from ..classes import Text, FoundEntity, OriginalEntity


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
            text_content = ground_truth_text.content

            print(f"Processing text: {text_content[:50]}...")

            # Perform Named Entity Recognition (NER)
            self.ner_handler.process_text(ground_truth_text)

            # Create a new Text object to hold the predicted entities
            predicted_text = Text(text_content)

            # Extract entities from NER and add them to the Text object
            found_entities = self.ner_handler.extract_entities(predicted_text)

            for entity in found_entities:
                predicted_text.add_entity(entity)

            # Perform Named Entity Disambiguation (NED)
            self.ned_handler.search_entities(predicted_text)

            # Extract entities for comparison
            predicted_entities = self.get_entities_from_text(predicted_text)
            ground_truth_entities = self.get_entities_from_text(ground_truth_text)

            print(f"Predicted entities: {predicted_entities}")
            print(f"Ground truth entities: {ground_truth_entities}")

            # Evaluate NER
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
        Extracts entities from a Text object, handling both FoundEntity and OriginalEntity instances.
        :param text: A Text object containing entities.
        :return: A list of dictionaries representing entities.
        """
        def map_entity(e):
            # Check the class type and map fields accordingly
            if isinstance(e, FoundEntity):
                return {
                    "entity_label": e.entity_label,
                    "start_position": e.start_position,
                    "end_position": e.end_position,
                    "uri": e.uri if e.uri is not None else None,
                }
            elif isinstance(e, OriginalEntity):
                return {
                    "entity_label": e.surface_form,
                    "start_position": e.position_start,
                    "end_position": e.position_end,
                    "dbpedia_uri": e.dbpedia_uri if e.dbpedia_uri is not None else None,
                    "wikidata_uri": e.wikidata_uri if e.wikidata_uri is not None else None,
                }
            else:
                raise TypeError(f"Unsupported entity type: {type(e)}")

        return [map_entity(e) for e in text.entities]
