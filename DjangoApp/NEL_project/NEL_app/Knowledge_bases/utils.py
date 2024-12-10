from ..NED_utlis.DBpedia.utils import *
from ..NED_utlis.Wikidata.utils import *
from ..models import *
from ..NER_utils.utils import *


def search_entities(sentence, text_obj, knowledge_base):
    print("Searching in knowledge_base:" + str(knowledge_base))
    ner_results = []

    # Process entities recognized in the sentence
    for entity in sentence.get_spans("ner"):

        entity_label = entity.text
        entity_type = entity.get_label("ner").value
        entity_probabilities = extract_entity_probabilities(entity)

        if knowledge_base == "dbpedia":
            best_result = search_dbpedia(entity_label)
            print(f"best_result: {best_result}")
            ner_results.append({
                "label": entity_label,
                "start": entity.start_position,
                "end": entity.end_position,
                "entity_type": entity_type,
                "uri": best_result["URI"] if best_result else "",
                "probabilities": entity_probabilities
            })

        else:
            best_result = search_wikidata(entity_label)
            take_only_first_result = best_result[0]
            ner_results.append({
                "label": entity_label,
                "start": entity.start_position,
                "end": entity.end_position,
                "entity_type": entity_type,
                "uri": take_only_first_result["URL"] if take_only_first_result else "",
                "probabilities": entity_probabilities
            })


    # Save all entities in the database
    save_entities_in_database(ner_results, text_obj)


def save_entities_in_database(ner_results, text_obj):
    for result_entity in ner_results:
        Entity.objects.create(
            text=text_obj,  # Associate with the Text object
            entity_label=result_entity["label"],
            entity_type=result_entity["entity_type"],
            start_position=result_entity["start"],
            end_position=result_entity["end"],
            uri=result_entity["uri"],
            probabilities=result_entity["probabilities"]  # Save probabilities
        )

