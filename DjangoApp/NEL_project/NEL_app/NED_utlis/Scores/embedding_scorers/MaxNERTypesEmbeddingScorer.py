import numpy as np

from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.BaseTypesEmbeddingScorer import BaseTypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig

class MaxNERTypesEmbeddingScorer(BaseTypesEmbeddingScorer):
    """
    Class to calculate a score for candidates based on the maximum similarity between any NER type and any ontology type.
    """
    def calculate_score(self, entity: Entity, ner_config: NERConfig):
        """
        Calculates scores for candidates based on the sum of (maximum similarity * NER probability).
        """
        if not entity.candidates or not entity.probabilities:
            return

        valid_ner_probabilities = [(cls, prob) for cls, prob in entity.probabilities if cls in ner_config.classes_definitions and cls != "O"]
        if not valid_ner_probabilities:
            for candidate in entity.candidates:
                candidate.score_maxner_types_embedding = 0.0
            return

        ner_classes = [item[0] for item in valid_ner_probabilities]
        ner_probabilities = np.array([float(item[1]) for item in valid_ner_probabilities])

        for candidate in entity.candidates:
            ontology_types = candidate.ontology_types
            if not ontology_types:
                candidate.score_maxner_types_embedding = 0.0
                continue

            similarity_matrix = self._calculate_similarity_matrix(ner_classes, ontology_types, ner_config)
            max_similarities = np.max(similarity_matrix, axis=1)
            score = float(np.sum(max_similarities * ner_probabilities)) if max_similarities.size > 0 else 0.0
            candidate.score_maxner_types_embedding = round(score, self.round_to_decimal_places)