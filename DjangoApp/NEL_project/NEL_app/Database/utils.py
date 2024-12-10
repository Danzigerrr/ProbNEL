from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.utils import *
from ..models import *
from ..NER_utils.utils import *


def create_entities_in_database(sentence, text_obj):
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
