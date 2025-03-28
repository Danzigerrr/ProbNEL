from .DBpedia.DBpediaSearch import DBpediaSearch
from .Wikidata.WikidataSearch import WikidataSearch
from ..Models.Entity import Entity
from .Scores.EntityCandidateScorer import EntityCandidateScorer
from .Candidate_Selector.CandidateSelector import CandidateSelector
from ..Models.Text import Text
from ..NER_utils.NERConfig import NERConfig


class NEDHandler:
    ALLOWED_KNOWLEDGE_BASES = ["dbpedia", "wikidata"]
    ALLOWED_CANDIDATE_SELECTION_STRATEGIES = ["sum_of_metrics", "candidate_selector_neural_network"]

    def __init__(self,
                 ner_config: NERConfig,
                 knowledge_base: str = ALLOWED_KNOWLEDGE_BASES[0],
                 candidate_selection_strategy: str = ALLOWED_CANDIDATE_SELECTION_STRATEGIES[0],
                 use_types_score: bool = True):
        """
        Initializes the NEDHandler with a specified knowledge base.
        :param knowledge_base: Either 'dbpedia' or 'wikidata'.
        """
        if knowledge_base.lower() not in self.ALLOWED_KNOWLEDGE_BASES:
            raise ValueError(f"Error: knowledge_base value must be on of the options: {self.ALLOWED_KNOWLEDGE_BASES}")
        self.knowledge_base = knowledge_base.lower()
        self.DBPediaSearch = DBpediaSearch()
        self.WikidataSearch = WikidataSearch()
        self.EntityCandidateScorer = EntityCandidateScorer()
        self.CandidateSelector = CandidateSelector()
        if candidate_selection_strategy.lower() not in self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES:
            raise ValueError(f"Error: candidate_selection_strategy value must be on of the options: {self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES}")
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_types_score = use_types_score
        self.ner_config = ner_config

    def perform_ned(self, text: Text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text: A Text object to associate with found entities.
        :return: None. Updates the text (Text instance) with found entities.
        """
        print(f"Searching in knowledge_base: {self.knowledge_base}")

        for entity in text.entities:  # Iterate over found entities
            entity_label = entity.entity_label  # Use the FoundEntity object for labels

            if self.knowledge_base == self.ALLOWED_KNOWLEDGE_BASES[0]:
                entity.candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label, 10)

            elif self.knowledge_base == self.ALLOWED_KNOWLEDGE_BASES[1]:
                entity.candidates = self.WikidataSearch.search_by_entity_surface_form(entity_label)

            self.select_best_candidate_for_entity(text, entity)

        return text

    def select_best_candidate_for_entity(self, text: Text, entity: Entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        self.EntityCandidateScorer.calculate_scores_for_candidates(text, entity, self.ner_config)

        if not self.use_types_score:
            for candidate in entity.candidates: candidate.score_types_embeddings_similarity = 0.0

        if entity.candidates:
            if self.candidate_selection_strategy == self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES[0]:
                entity.candidates.sort(key=lambda x: x.score_final, reverse=True)
                entity.best_candidate_uri = entity.candidates[0].uri
            elif self.candidate_selection_strategy == self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES[1]:
                self.CandidateSelector.select_best_candidate_for_entity(entity=entity)

    def print_top_candidates(self, entity: Entity, top_n_candidates_to_print = 3):
        print(f"\n### Best candidate(s) for entity '{entity.entity_label}'({entity.start_position}, {entity.end_position}):")
        for candidate in entity.candidates[:top_n_candidates_to_print]:
            candidate.print_details()

