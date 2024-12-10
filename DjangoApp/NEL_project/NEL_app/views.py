from django.shortcuts import render
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from .Database.utils import *

print("Loading model...")
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
print("Model loaded.")


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
            create_entities_in_database(sentence, text_obj)

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