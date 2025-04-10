from .DBpedia.DBpediaSearch import DBpediaSearch
from .Wikidata.WikidataSearch import WikidataSearch
from ..model_components.Entity import Entity
from .Scores.EntityCandidateScorer import EntityCandidateScorer
from .Candidate_Selector.CandidateSelector import CandidateSelector
from ..model_components.Text import Text
from ..NER_utils.NERConfig import NERConfig

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NEDHandler:
    ALLOWED_KNOWLEDGE_BASES = ["dbpedia", "wikidata"]
    ALLOWED_CANDIDATE_SELECTION_STRATEGIES = ["sum_of_metrics",
                                              "candidate_selector_neural_network",
                                              "candidate_selector_random_forest_classifier",
                                              "candidate_selector_svm"
                                              ]

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
        self.EntityCandidateScorer = EntityCandidateScorer()
        if candidate_selection_strategy not in self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES:
            raise ValueError(f"Error: candidate_selection_strategy value must be on of the options: {self.ALLOWED_CANDIDATE_SELECTION_STRATEGIES}")
        self.CandidateSelector = CandidateSelector(candidate_selection_strategy, use_types_score)
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_types_score = use_types_score
        self.ner_config = ner_config

    def perform_ned(self, text: Text):
        """
        Searches for entities in the given text using the specified knowledge base.
        :param text: A Text object to associate with found entities.
        :return: None. Updates the text (Text instance) with found entities.
        """
        # print(f"Searching in knowledge_base: {self.knowledge_base}")

        for entity in text.entities:  # Iterate over found entities
            entity_label = entity.entity_label  # Use the FoundEntity object for labels

            if self.knowledge_base == self.ALLOWED_KNOWLEDGE_BASES[0]:
                entity.candidates = self.DBPediaSearch.search_by_entity_surface_form(entity_label, 10)


            self.select_best_candidate_for_entity(text, entity)

        return text

    def select_best_candidate_for_entity(self, text: Text, entity: Entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        self.EntityCandidateScorer.calculate_scores_for_candidates(text, entity, self.ner_config, self.use_types_score)

        # sort candidates in the candidates list
        entity.candidates.sort(key=lambda x: x.score_final, reverse=True)

        if entity.candidates:
                self.CandidateSelector.model.select_best_candidate_for_entity(entity=entity)

    def print_top_candidates(self, entity: Entity, top_n_candidates_to_print = 3):
        print(f"\n### Best candidate(s) for entity '{entity.entity_label}'({entity.start_position}, {entity.end_position}):")
        for candidate in entity.candidates[:top_n_candidates_to_print]:
            candidate.print_details()

