from ..NED_utlis.DBpedia.utils import *
from ..NED_utlis.Wikidata.utils import *
from ..classes import Text, FoundEntity
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
            # print(f"best_result: {best_result}")
            ner_results.append(FoundEntity(
                text=text_obj,
                entity_label=entity_label,
                entity_type=entity_type,
                start_position=entity.start_position,
                end_position=entity.end_position,
                uri=best_result["URI"] if best_result else "",
                probabilities=entity_probabilities
            ))

        else:
            best_result = search_wikidata(entity_label)
            take_only_first_result = best_result[0]
            ner_results.append(FoundEntity(
                text=text_obj,
                entity_label=entity_label,
                entity_type=entity_type,
                start_position=entity.start_position,
                end_position=entity.end_position,
                uri=take_only_first_result["URL"] if take_only_first_result else "",
                probabilities=entity_probabilities
            ))

    # Add entities to the text object
    add_entities_to_text(ner_results, text_obj)


def add_entities_to_text(ner_results, text_obj):
    """Add recognized entities to the given text object."""
    for entity in ner_results:
        text_obj.add_entity(entity)
