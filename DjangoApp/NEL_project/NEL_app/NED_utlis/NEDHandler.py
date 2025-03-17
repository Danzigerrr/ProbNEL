from .DBpedia.utils import search_dbpedia_by_entity_surface_form
from .Wikidata.utils import search_wikidata
from ..NER_utils import NERHandler
from ..classes import Text, Entity
import json
from .DBpedia.DBpedia_classes import *


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
                search_results = search_dbpedia_by_entity_surface_form(entity_label)
                entity.candidates = self.format_candidates_list(search_results)
                self.choose_best_candidate_for_entity(entity)
            elif self.knowledge_base == "wikidata":
                best_result = search_wikidata(entity_label)
                entity.wikidata_uri = best_result.get("URI") if best_result else ""

    def format_candidates_list(self, search_results):
        """
        Extract the best result from the DBpedia Lookup API response.

        :param search_results: The JSON response from the DBpedia Lookup API.
        :return: A list of Candidate objects or None if no valid results are found.
        """
        candidates = []
        if search_results and search_results.get("docs"):
            for doc in search_results["docs"]:
                label = doc.get("label", [""])[0]
                ontology_types = doc.get("type", [])
                comment = doc.get("comment", [""])[0]
                uri = doc.get("resource", [""])[0]

                candidate = Candidate(
                    label=label,
                    ontology_types=ontology_types,
                    comment=comment,
                    uri=uri,
                    score_ner_to_ontology=None
                )
                candidates.append(candidate)
        return candidates

    def choose_best_candidate_for_entity(self, entity):
        entity.dbpedia_uri = entity.candidates[0].uri
