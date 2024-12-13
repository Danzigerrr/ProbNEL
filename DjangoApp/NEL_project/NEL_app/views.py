from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from .Knowledge_bases.utils import *
import json
from django.views.decorators.csrf import csrf_exempt
from .testing.utils import *

print("Loading model...")
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
print("Model loaded.")


def index(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")
        source = request.POST.get("source", "")  # Retrieve the source parameter

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            # Create or get Text object
            text_obj = Text(user_input)

            # Process the sentence with the Flair model
            sentence = Sentence(user_input)
            tagger.predict(sentence, return_probabilities_for_all_classes=True)

            # Determine the source-specific processing
            if source == "dbpedia":
                search_entities(sentence, text_obj, knowledge_base="dbpedia")
            elif source == "wikidata":
                search_entities(sentence, text_obj, knowledge_base="wikidata")
            else:
                return JsonResponse({"error": "Invalid source specified."}, status=400)

            # Collect the entities associated with the text
            entities = [
                {
                    "entity_label": e.entity_label,
                    "entity_type": e.entity_type,
                    "start_position": e.start_position,
                    "end_position": e.end_position,
                    "uri": e.uri,
                    "probabilities": e.probabilities,
                }
                for e in text_obj.entities
            ]

            # Return both text and entities as a response
            return JsonResponse({
                "text": text_obj.content,
                "entities": entities
            })

        except Exception as e:
            return JsonResponse({"error": f"Error processing input: {str(e)}"}, status=500)

    # For GET requests, render the template
    return render(request, "NEL_app/index.html")


@csrf_exempt
def upload_dataset(request):
    if request.method == "POST" and request.FILES.get("dataset"):
        try:
            dataset_file = request.FILES["dataset"]

            dataset_content = json.load(dataset_file)

            dataset = parse_dataset_content(dataset_content, dataset_file.name)

            print_parsing_info(dataset)

            evaluation_results = run_test_on_dataset(dataset)

            # Return the serialized object as JSON
            return JsonResponse({
                "success": True,
                "evaluation_results": json.loads(evaluation_results.to_json())
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)


def run_test_on_dataset(dataset):
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
