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

            return JsonResponse({"success": True, "message": "Dataset uploaded successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

