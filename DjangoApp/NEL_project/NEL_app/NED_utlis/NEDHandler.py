from .DBpedia.DBpedia_search import DBPediaSearch
from .Wikidata.utils import search_wikidata
from ..NER_utils import NERHandler
from ..classes import Text, Entity
import json
from .DBpedia.DBpedia_classes import *
from .Scores.NER_type_to_Ontology_mapping_score import EntityCandidateScorer


class NEDHandler:
    def __init__(self, knowledge_base: str = "dbpedia"):
        """
        Initializes the NEDHandler with a specified knowledge base.
        :param knowledge_base: Either 'dbpedia' or 'wikidata'.
        """
        if knowledge_base.lower() not in ["dbpedia", "wikidata"]:
            raise ValueError("Knowledge base must be 'dbpedia' or 'wikidata'")
        self.knowledge_base = knowledge_base.lower()
        self.DBPediaSearch = DBPediaSearch()
        self.EntityCandidateScorer = EntityCandidateScorer()

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
                entity.candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label)
                self.choose_best_candidate_for_entity(entity)
            elif self.knowledge_base == "wikidata":
                best_result = search_wikidata(entity_label)
                entity.wikidata_uri = best_result.get("URI") if best_result else ""

    def choose_best_candidate_for_entity(self, entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        self.EntityCandidateScorer.calculate_candidate_scores(entity)
        if entity.candidates:
            entity.candidates.sort(key=lambda x: x.candidate_score, reverse=False)
            entity.dbpedia_uri = entity.candidates[0].uri
        entity.dbpedia_uri = entity.candidates[0].uri
