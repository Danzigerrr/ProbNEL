from django.shortcuts import render
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .testing.utils import *
import json
from .NED_utlis.NEDHandler import NEDHandler
from .NER_utils.NERHandler import NERHandler
from .Evaluation.EvaluationHandler import EvaluationHandler


def index(request: HttpRequest) -> HttpResponse:
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")
        knowledge_graph = request.POST.get("knowledge_graph", "")  # Retrieve the knowledge_graph parameter

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            text_obj = Text(user_input)

            ner = NERHandler("flair/ner-english-ontonotes-fast")
            ner.perform_ner(text_obj)

            if knowledge_graph in ["dbpedia", "wikidata"]:
                ned = NEDHandler(knowledge_graph)
                ned.perform_ned(text_obj)
            else:
                return JsonResponse({"error": "Invalid knowledge_graph specified. Allowed values: dbpedia, wikidata"}, status=400)

            json_response = create_json_response(text_obj)
            return json_response

        except Exception as e:
            return JsonResponse({"error": f"Error processing input: {str(e)}"}, status=500)

    # For GET requests, render the template
    return render(request, "NEL_app/index.html")


def create_json_response(text_obj: Text):
    entities = [
        {
            "entity_label": e.entity_label,
            "entity_type": e.entity_type,
            "start_position": e.start_position,
            "end_position": e.end_position,
            "dbpedia_uri": e.dbpedia_uri,
            "wikidata_uri": e.wikidata_uri,
            "probabilities": e.probabilities,
        }
        for e in text_obj.entities
    ]

    return JsonResponse({
        "text": text_obj.content,
        "entities": entities
    })


@csrf_exempt
def run_test_on_dataset(request: HttpRequest) -> HttpResponse:
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

            serialised_evaluation_results = serialize_the_evaluation_results_to_json(evaluation_results)
            return serialised_evaluation_results

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)


def serialize_the_evaluation_results_to_json(evaluation_results):
    # Serialize the evaluation results to JSON
    ner_results_json = json.loads(evaluation_results[0].to_json())
    ned_results_json = json.loads(evaluation_results[1].to_json())
    # Return the serialized evaluation results
    serialised_evaluation_results = JsonResponse({
        "success": True,
        "evaluation_results": {
            "ner": ner_results_json,
            "ned": ned_results_json
        }
    })
    return serialised_evaluation_results

