from .DBpedia.utils import search_dbpedia
from .Wikidata.utils import search_wikidata
from ..NER_utils import NERHandler
from ..classes import Text, Entity


class NEDHandler:
    def __init__(self, knowledge_base: str = "dbpedia"):
        """
        Initializes the NEDHandler with a specified knowledge base.
        :param knowledge_base: Either 'dbpedia' or 'wikidata'.
        """
        if knowledge_base.lower() not in ["dbpedia", "wikidata"]:
            raise ValueError("Knowledge base must be 'dbpedia' or 'wikidata'")
        self.knowledge_base = knowledge_base.lower()

    def perform_ned(self, text_obj: Text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text_with_ner_tags: Text annotated with NER tags.
        :param text_obj: A Text object to associate with found entities.
        :return: None. Updates the text_obj with found entities.
        """
        print(f"Searching in knowledge_base: {self.knowledge_base}")

        for entity in text_obj.entities:  # Iterate over found entities
            entity_label = entity.entity_label  # Use the FoundEntity object for labels

            if self.knowledge_base == "dbpedia":
                best_result = search_dbpedia(entity_label)
                entity.dbpedia_uri = best_result.get("URI") if best_result else ""
            elif self.knowledge_base == "wikidata":
                best_result = search_wikidata(entity_label)
                entity.wikidata_uri = best_result.get("URI") if best_result else ""

    def add_entities_to_text(self, ner_results: list, text_obj: Text):
        """
        Adds recognized entities to the given Text object.
        :param ner_results: List of FoundEntity objects.
        :param text_obj: A Text object to update with entities.
        :return: None.
        """
        for entity in ner_results:
            text_obj.add_entity(entity)
