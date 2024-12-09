from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger

from .models import *
from .NER_utils.utils import *
from .NED_utlis.utils import *

# Load the Flair NER model once (using the 'fast' version)
print("Loading model...")
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
print("Model loaded.")

# Global variables
sentence = None


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