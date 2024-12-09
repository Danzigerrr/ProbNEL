from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
import requests

from .models import *
from .NER_utils.utils import *

# Load the Flair NER model once (using the 'fast' version)
print("Loading model...")
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
print("Model loaded.")

# Global variables
DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"
sentence = None

# DBpedia lookup function
def search_dbpedia(entity_text, dbpedia_type=None, max_results=3):
    params = {
        "query": entity_text,
        "format": "JSON",
        "maxResults": max_results,
    }
    if dbpedia_type:
        params["typeName"] = dbpedia_type
        params["typeNameRequired"] = "true"

    try:
        response = requests.get(DBPEDIA_LOOKUP_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("docs"):
            best_doc = max(data["docs"], key=lambda doc: float(doc.get("score", [0])[0]))
            return {
                "Label": best_doc.get("label", ["Unknown"])[0],
                "URI": best_doc.get("resource", [""])[0],
                "Description": best_doc.get("comment", ["No description available"])[0],
                "Score": float(best_doc.get("score", [0])[0]),
            }
    except requests.exceptions.RequestException as e:
        print(f"Error querying DBpedia: {e}")

    return None

def get_entities_and_links(sentence, text_obj):
    ner_results = []

    for entity in sentence.get_spans("ner"):
        entity_text = entity.text
        entity_type = entity.get_label("ner").value
        best_result = search_dbpedia(entity_text)
        ner_results.append({
            "text": entity_text,
            "start": entity.start_position,
            "end": entity.end_position,
            "entity_group": entity_type,
            "uri": best_result["URI"] if best_result else "",
        })

        # Save the entity to the database, associating it with the text_obj
        Entity.objects.create(
            text=text_obj,  # Associate with the Text object
            entity_text=entity_text,
            entity_type=entity_type,
            start_position=entity.start_position,
            end_position=entity.end_position,
            uri=best_result["URI"] if best_result else "",
            probabilities=extract_entity_probabilities(entity)  # Get probabilities
        )

    return ner_results


def index(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        user_input = request.POST.get("user_input", "")

        if not user_input:
            return JsonResponse({"error": "Input text is required."}, status=400)

        try:
            # Create Text object
            text_obj = Text.objects.create(content=user_input)

            # Process the sentence with the Flair model
            sentence = Sentence(user_input)
            tagger.predict(sentence, return_probabilities_for_all_classes=True)

            # Pass the text_obj to get_entities_and_links
            ner_results = get_entities_and_links(sentence, text_obj)

            # Collect the entities associated with the text
            entities = Entity.objects.filter(text=text_obj).values('entity_text', 'entity_type', 'start_position', 'end_position', 'uri', 'probabilities')

            # Return both text and entities as a response
            return JsonResponse({
                "text": text_obj.content,
                "entities": list(entities)
            })

        except Exception as e:
            return JsonResponse({"error": f"Error processing input: {str(e)}"}, status=500)

    # For GET requests, render the template
    return render(request, "NEL_app/index.html")