from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from django.views.decorators.csrf import csrf_exempt
from .testing.utils import *
import json
from .Evaluation.utils import run_test_on_dataset
from .NED_utlis.NEDHandler import NEDHandler

ner_tagger_model_name = "flair/ner-english-ontonotes-fast"
print(f"Loading NER model: {ner_tagger_model_name}")
tagger = SequenceTagger.load(ner_tagger_model_name)
print("Model NER loaded.")


def index(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")
        knowledge_graph = request.POST.get("knowledge_graph", "")  # Retrieve the knowledge_graph parameter

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            # Create or get Text object
            text_from_user = Text(user_input)

            # Process the sentence with the Flair model
            text_with_ner_tags = Sentence(user_input)
            tagger.predict(text_with_ner_tags, return_probabilities_for_all_classes=True)

            # Determine the knowledge_graph-specific processing
            if knowledge_graph == "dbpedia" or knowledge_graph == "wikidata":
                ned = NEDHandler(knowledge_graph)
                ned.search_entities(text_with_ner_tags, text_from_user)
            else:
                return JsonResponse({"error": "Invalid knowledge_graph for Knowledge Graph specified. "
                                              "Allowed values: dbpedia, wikidata"}, status=400)

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
                for e in text_from_user.entities
            ]

            # Return both text and entities as a response
            return JsonResponse({
                "text": text_from_user.content,
                "entities": entities
            })

        except Exception as e:
            return JsonResponse({"error": f"Error processing input: {str(e)}"}, status=500)

    # For GET requests, render the template
    return render(request, "NEL_app/index.html")


@csrf_exempt
def run_test_on_dataset(request):
    if request.method == "POST" and request.FILES.get("dataset"):
        try:
            dataset_file = request.FILES["dataset"]

            dataset_content = json.load(dataset_file)

            dataset = parse_dataset_content(dataset_content, dataset_file.name)

            print_parsing_info(dataset)

            evaluation_results = run_test_on_dataset(dataset, tagger)

            # Return the serialized object as JSON
            return JsonResponse({
                "success": True,
                "evaluation_results": json.loads(evaluation_results.to_json())
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

