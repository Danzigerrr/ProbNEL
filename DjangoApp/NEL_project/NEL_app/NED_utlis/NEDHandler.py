from .DBpedia.DBpediaSearch import DBpediaSearch
from .Wikidata.utils import search_wikidata
from ..classes import Text
from .Scores.EntityCandidateScorer import EntityCandidateScorer


class NEDHandler:
    def __init__(self, knowledge_base: str = "dbpedia"):
        """
        Initializes the NEDHandler with a specified knowledge base.
        :param knowledge_base: Either 'dbpedia' or 'wikidata'.
        """
        if knowledge_base.lower() not in ["dbpedia", "wikidata"]:
            raise ValueError("Knowledge base must be 'dbpedia' or 'wikidata'")
        self.knowledge_base = knowledge_base.lower()
        self.DBPediaSearch = DBpediaSearch()
        self.EntityCandidateScorer = EntityCandidateScorer()

    def perform_ned(self, text_obj: Text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text_obj: A Text object to associate with found entities.
        :return: None. Updates the text_obj with found entities.
        """
        print(f"Searching in knowledge_base: {self.knowledge_base}")

        for entity in text_obj.entities:  # Iterate over found entities
            entity_label = entity.entity_label  # Use the FoundEntity object for labels

            if self.knowledge_base == "dbpedia":
                entity.candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label)
                self.select_best_candidate_for_entity(entity)
            elif self.knowledge_base == "wikidata":
                best_result = search_wikidata(entity_label)
                entity.wikidata_uri = best_result.get("URI") if best_result else ""

    def select_best_candidate_for_entity(self, entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        self.EntityCandidateScorer.calculate_scores_for_candidates(entity)

        if entity.candidates:
            entity.candidates.sort(key=lambda x: x.candidate_score, reverse=True)
            entity.dbpedia_uri = entity.candidates[0].uri

            print(f"\n### Best candidates for entity {entity.entity_label}:")
            for candidate in entity.candidates[:3]:
                candidate.print_details()

