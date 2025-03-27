from .DBpedia.DBpediaSearch import DBpediaSearch
from .Wikidata.WikidataSearch import WikidataSearch
from ..Models.Text import Text
from ..Models.Entity import Entity
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
        self.WikidataSearch = WikidataSearch()
        self.EntityCandidateScorer = EntityCandidateScorer()

    def perform_ned(self, text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text: A Text object to associate with found entities.
        :return: None. Updates the text (Text instance) with found entities.
        """
        print(f"Searching in knowledge_base: {self.knowledge_base}")

        for entity in text.entities:  # Iterate over found entities
            entity_label = entity.entity_label  # Use the FoundEntity object for labels

            if self.knowledge_base == "dbpedia":
                entity.candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label, 10)

            elif self.knowledge_base == "wikidata":
                entity.candidates = self.WikidataSearch.search_by_entity_surface_form(entity_label)

            self.select_best_candidate_for_entity(text, entity)

    def select_best_candidate_for_entity(self, text, entity: Entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        self.EntityCandidateScorer.calculate_scores_for_candidates(text, entity)

        if entity.candidates:
            entity.candidates.sort(key=lambda x: x.score_final, reverse=True)
            entity.best_candidate_uri = entity.candidates[0].uri

            # self.print_top_candidates(entity)

    def print_top_candidates(self, entity, top_n_candidates_to_print = 3):
        print(f"\n### Best candidate(s) for entity '{entity.entity_label}'({entity.start_position}, {entity.end_position}):")
        for candidate in entity.candidates[:top_n_candidates_to_print]:
            candidate.print_details()

