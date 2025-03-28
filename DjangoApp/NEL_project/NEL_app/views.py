from django.shortcuts import render
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .Models.Text import Text
from .NED_utlis.NEDHandler import NEDHandler
from .NER_utils.NERHandler import NERHandler
from .Evaluation.EvaluationHandler import EvaluationHandler
from .Evaluation.DatasetLoader import DatasetLoader


def index(request: HttpRequest) -> HttpResponse:
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")
        knowledge_graph = request.POST.get("knowledge_graph", "")  # Retrieve the knowledge_graph parameter

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            text_obj = Text(user_input)

            ner = NERHandler()
            text_obj = ner.perform_ner(text_obj)

            if knowledge_graph in ["dbpedia", "wikidata"]:
                ned = NEDHandler(knowledge_graph)
                text_obj = ned.perform_ned(text_obj)
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
            "best_candidate_uri": e.best_candidate_uri,
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
            # Load dataset using DatasetLoader
            dataset_loader = DatasetLoader()
            dataset = dataset_loader.load_dataset(request.FILES["dataset"])

            # Print dataset information
            dataset_loader.print_dataset_info(dataset)

            # Initialize handlers
            ner_handler = NERHandler()
            ned_handler = NEDHandler("dbpedia")

            # Run evaluation
            evaluation_handler = EvaluationHandler(ner_handler, ned_handler)
            evaluation_results = evaluation_handler.run_test_on_dataset(dataset)
            evaluation_results.print_results()

            serialised_evaluation_results = serialize_the_evaluation_results_to_json(evaluation_results)

            return serialised_evaluation_results

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)


def serialize_the_evaluation_results_to_json(evaluation_results):
    serialised_evaluation_results = JsonResponse({
        "success": True,
        "evaluation_results": evaluation_results.to_json()
    })
    return serialised_evaluation_results

