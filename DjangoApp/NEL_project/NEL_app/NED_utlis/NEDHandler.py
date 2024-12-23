from .DBpedia.utils import search_dbpedia
from .Wikidata.utils import search_wikidata
from ..classes import Text, FoundEntity
from ..NER_utils.utils import extract_entity_probabilities


class NEDHandler:
    def __init__(self, knowledge_base: str = "dbpedia"):
        """
        Initializes the NEDHandler with a specified knowledge base.
        :param knowledge_base: Either 'dbpedia' or 'wikidata'.
        """
        if knowledge_base.lower() not in ["dbpedia", "wikidata"]:
            raise ValueError("Knowledge base must be 'dbpedia' or 'wikidata'")
        self.knowledge_base = knowledge_base.lower()

    def search_entities(self, text_with_ner_tags, text_obj: Text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text_with_ner_tags: Text annotated with NER tags.
        :param text_obj: A Text object to associate with found entities.
        :return: None. Updates the text_obj with found entities.
        """
        print(f"Searching in knowledge_base: {self.knowledge_base}")
        ner_results = []

        for entity in text_with_ner_tags.get_spans("ner"):
            entity_label = entity.text
            entity_type = entity.get_label("ner").value
            entity_probabilities = extract_entity_probabilities(entity)

            if self.knowledge_base == "dbpedia":
                best_result = search_dbpedia(entity_label)
                ner_results.append(self.create_found_entity(entity, text_obj, best_result.get("URI"), entity_probabilities))
            elif self.knowledge_base == "wikidata":
                best_result = search_wikidata(entity_label)
                first_result = best_result[0] if best_result else None
                ner_results.append(self.create_found_entity(entity, text_obj, first_result.get("URL") if first_result else "", entity_probabilities))

        # Add entities to the text object
        self.add_entities_to_text(ner_results, text_obj)

    def create_found_entity(self, entity, text_obj: Text, uri: str, probabilities: dict) -> FoundEntity:
        """
        Creates a FoundEntity object from the given data.
        :param entity: NER entity object.
        :param text_obj: Associated Text object.
        :param uri: The URI or URL of the entity.
        :param probabilities: Probabilities extracted for the entity.
        :return: A FoundEntity object.
        """
        return FoundEntity(
            text=text_obj,
            entity_label=entity.text,
            entity_type=entity.get_label("ner").value,
            start_position=entity.start_position,
            end_position=entity.end_position,
            uri=uri,
            probabilities=probabilities
        )

    def add_entities_to_text(self, ner_results: list, text_obj: Text):
        """
        Adds recognized entities to the given Text object.
        :param ner_results: List of FoundEntity objects.
        :param text_obj: A Text object to update with entities.
        :return: None.
        """
        for entity in ner_results:
            text_obj.add_entity(entity)
