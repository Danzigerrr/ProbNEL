from .EvaluationResults import EvaluationResults
from flair.data import Sentence
from ..classes import Text
from ..Knowledge_bases.utils import search_entities


def run_test_on_dataset(dataset, tagger):
    """Run NER and NED on the dataset and evaluate prediction accuracy."""
    evaluation_results = EvaluationResults()

    # Iterate over all Text objects in memory
    for ground_truth_text in dataset.texts:
        print("NER - processing: " + str(ground_truth_text.content[:50]))

        # Process the text using the Flair model for NER
        sentence = Sentence(ground_truth_text.content)
        tagger.predict(sentence, return_probabilities_for_all_classes=True)

        print("Run NED using DBpedia knowledge base")
        predicted_text = Text(ground_truth_text.content)
        search_entities(sentence, predicted_text, knowledge_base="dbpedia")

        print("Collect entities from in-memory objects associated with the current text")
        predicted_entities = get_entities_from_text(predicted_text)
        ground_truth_entities = get_entities_from_text(ground_truth_text)

        print(f"Predicted entities: {predicted_entities}")
        print("------\n")
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

    # Finalize and print evaluation results
    evaluation_results.finalize()
    evaluation_results.print()

    return evaluation_results


def get_entities_from_text(text):
    entities = [
        {
            "entity_label": e.entity_label,
            "start_position": e.start_position,
            "end_position": e.end_position,
            "uri": e.uri,
        }
        for e in text.entities
    ]
    return entities
