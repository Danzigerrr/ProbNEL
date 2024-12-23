from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .testing.utils import *
import json
from .NED_utlis.NEDHandler import NEDHandler
from .NER_utils.NERHandler import NERHandler
from .Evaluation.EvaluationHandler import EvaluationHandler


def index(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")
        knowledge_graph = request.POST.get("knowledge_graph", "")  # Retrieve the knowledge_graph parameter

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            # Create or get Text object
            text_obj = Text(user_input)

            # Process the sentence with the Flair model
            # Initialize NERHandler and process the text
            ner = NERHandler("flair/ner-english-ontonotes-fast")
            ner.process_text(text_obj)

            # Determine the knowledge_graph-specific processing
            if knowledge_graph in ["dbpedia", "wikidata"]:
                ned_handler = NEDHandler(knowledge_graph)
                ned_handler.search_entities(text_obj)
            else:
                return JsonResponse({"error": "Invalid knowledge_graph specified. Allowed values: dbpedia, wikidata"}, status=400)

            entities = get_found_entities(text_obj)

            # Return both text and entities as a response
            return JsonResponse({
                "text": text_obj.content,
                "entities": entities
            })

        except Exception as e:
            return JsonResponse({"error": f"Error processing input: {str(e)}"}, status=500)

    # For GET requests, render the template
    return render(request, "NEL_app/index.html")


def get_found_entities(text_from_user):
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
    return entities


@csrf_exempt
def run_test_on_dataset(request):
    if request.method == "POST" and request.FILES.get("dataset"):
        try:
            dataset_file = request.FILES["dataset"]
            dataset_content = json.load(dataset_file)
            dataset = parse_dataset_content(dataset_content, dataset_file.name)

            print_parsing_info(dataset)

            # Initialize handlers
            ner_handler = NERHandler("flair/ner-english-ontonotes-fast")
            ned_handler = NEDHandler("dbpedia")
            evaluation_handler = EvaluationHandler(ner_handler, ned_handler)

            # Run evaluation
            evaluation_results = evaluation_handler.run_test_on_dataset(dataset)

            # Return the serialized evaluation results
            return JsonResponse({
                "success": True,
                "evaluation_results": json.loads(evaluation_results.to_json())
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

