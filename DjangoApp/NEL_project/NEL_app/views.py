from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from .Knowledge_bases.utils import *
import json
from django.views.decorators.csrf import csrf_exempt
from .testing.Dataset import *
from .testing.Text import *
from .testing.EntityMention import *


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

            # Initialize the Dataset object
            dataset = Dataset()

            # Parse each text and its entity mentions
            for text_entry in dataset_content:
                text_content = text_entry["text"]
                entity_mentions = []

                for mention in text_entry["entity_mentions"]:
                    entity_mention = EntityMention(
                        surface_form=mention["surface_form"],
                        position_start=mention["position"]["start"],
                        position_end=mention["position"]["end"],
                        dbpedia_uri=mention["dbpedia_target_uri"],
                        wikidata_uri=mention["wikidata_target_uri"]
                    )
                    entity_mentions.append(entity_mention)

                # Create a Text object and add it to the Dataset
                text_object = Text(text=text_content, entity_mentions=entity_mentions)
                dataset.add_text(text_object)

            # Debug: Print the parsed dataset structure
            for text_obj in dataset.texts:
                print(f"Text: {text_obj.text}")
                for mention in text_obj.entity_mentions:
                    print(f"  Entity: {mention.surface_form}, Start: {mention.position_start}, "
                          f"End: {mention.position_end}, DBpedia: {mention.dbpedia_uri}, "
                          f"Wikidata: {mention.wikidata_uri}")

            return JsonResponse({"success": True, "message": "Dataset uploaded successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)
