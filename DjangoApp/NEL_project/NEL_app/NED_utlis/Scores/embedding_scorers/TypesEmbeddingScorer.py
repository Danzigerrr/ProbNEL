import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.BaseTypesEmbeddingScorer import BaseTypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


class TypesEmbeddingScorer(BaseTypesEmbeddingScorer):
    """
    Class to calculate a weighted average score for candidates based on NER entity type and candidate ontology types.
    """
    def calculate_score(self, entity: Entity, ner_config: NERConfig):
        """
        Calculates scores for candidates of an entity based on weighted average of NER type - ontology type similarities.
        """
        if not entity.candidates or not entity.probabilities:
            return

        valid_ner_probabilities = [(cls, prob) for cls, prob in entity.probabilities if cls in ner_config.classes_definitions and cls != "O"]
        if not valid_ner_probabilities:
            for candidate in entity.candidates:
                candidate.score_basic_types_embedding = 0.0
            return

        ner_classes = [item[0] for item in valid_ner_probabilities]
        ner_probabilities = np.array([float(item[1]) for item in valid_ner_probabilities])

        for candidate in entity.candidates:
            ontology_types = candidate.ontology_types
            if not ontology_types:
                candidate.score_basic_types_embedding = 0.0
                continue

            similarity_matrix = self._calculate_similarity_matrix(ner_classes, ontology_types, ner_config)

            weighted_similarities = np.array([
                np.sum(ner_probabilities * similarity_matrix[:, j]) for j in range(len(ontology_types))
            ])
            score = float(np.mean(weighted_similarities)) if weighted_similarities.size > 0 else 0.0
            candidate.score_basic_types_embedding = round(score, self.round_to_decimal_places)
