from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from .Knowledge_bases.utils import *
import json
from django.views.decorators.csrf import csrf_exempt
from .testing.utils import *
from django.db.models import Q


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
            # Create Text object
            text_obj = Text.objects.create(content=user_input)

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
            entities = Entity.objects.filter(text=text_obj).values('entity_label', 'entity_type', 'start_position',
                                                                   'end_position', 'uri', 'probabilities')

            # Return both text and entities as a response
            return JsonResponse({
                "text": text_obj.content,
                "entities": list(entities)
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

            dataset = parse_dataset_content(dataset_content)

            print_parsing_info(dataset)

            run_test_on_dataset(dataset)

            return JsonResponse({"success": True, "message": "Dataset uploaded successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)


def run_test_on_dataset(dataset):
    """Run NER and NED on the dataset and evaluate prediction accuracy."""
    correct_predictions = 0
    total_predictions = 0
    correct_uri_matches = 0
    total_uris = 0

    for text_obj in dataset.texts:
        print("NER - processing: " + str(text_obj))
        # Process the text using the Flair model for NER
        sentence = Sentence(text_obj.text)
        tagger.predict(sentence, return_probabilities_for_all_classes=True)

        print("Run NED using DBpedia knowledge base")
        search_entities(sentence, text_obj, knowledge_base="dbpedia")

        print("Collect entities from the database associated with the current text")
        predicted_entities = Entity.objects.filter(text=text_obj).values(
            "entity_label", "start_position", "end_position", "uri"
        )

        # Compare predicted entities with ground-truth entity mentions
        for mention in text_obj.entity_mentions:
            total_predictions += 1

            # Check if a predicted entity matches the ground truth
            matching_entity = next(
                (
                    pred
                    for pred in predicted_entities
                    if pred["entity_label"] == mention.surface_form
                       and pred["start_position"] == mention.position_start
                       and pred["end_position"] == mention.position_end
                ),
                None,
            )

            if matching_entity:
                correct_predictions += 1
                total_uris += 1

                # Check if the predicted URI matches the ground-truth URI
                if matching_entity["uri"] == mention.dbpedia_uri:
                    correct_uri_matches += 1

    # Calculate accuracy for NER and NED
    ner_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    ned_accuracy = (correct_uri_matches / total_uris) * 100 if total_uris > 0 else 0

    print("NER and NED Evaluation:")
    print(f"Total ground-truth mentions: {total_predictions}")
    print(f"Correctly predicted mentions (NER): {correct_predictions}")
    print(f"NER Accuracy: {ner_accuracy:.2f}%")
    print(f"Total ground-truth URIs: {total_uris}")
    print(f"Correctly matched URIs (NED): {correct_uri_matches}")
    print(f"NED Accuracy: {ned_accuracy:.2f}%")
