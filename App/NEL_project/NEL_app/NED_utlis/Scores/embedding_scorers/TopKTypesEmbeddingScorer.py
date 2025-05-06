import numpy as np

from App.NEL_project.NEL_app.Models.Entity import Entity
from App.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.BaseTypesEmbeddingScorer import \
    BaseTypesEmbeddingScorer
from App.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig



class TopKTypesEmbeddingScorer(BaseTypesEmbeddingScorer):
    """
    Class to calculate a score for candidates based on the top-k NER entity types and candidate ontology types.
    """
    def __init__(self, top_k=3, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def calculate_score(self, entity: Entity, ner_config: NERConfig):
        """
        Calculates scores for candidates based on the maximum weighted similarity of the top-k NER types with ontology types.
        """
        if not entity.candidates or not entity.probabilities:
            return

        sorted_ner_probabilities = sorted(
            [(cls, prob) for cls, prob in entity.probabilities if cls in ner_config.classes_definitions and cls != "O"],
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        if not sorted_ner_probabilities:
            for candidate in entity.candidates:
                candidate.score_topk_types_embedding = 0.0
            return

        ner_classes = [item[0] for item in sorted_ner_probabilities]
        ner_probabilities = np.array([float(item[1]) for item in sorted_ner_probabilities])

        for candidate in entity.candidates:
            ontology_types = candidate.ontology_types
            if not ontology_types:
                candidate.score_topk_types_embedding = 0.0
                continue

            similarity_matrix = self._calculate_similarity_matrix(ner_classes, ontology_types, ner_config)

            weighted_similarities = np.array([
                np.sum(ner_probabilities * similarity_matrix[:, j]) for j in range(len(ontology_types))
            ])
            score = float(np.max(weighted_similarities)) if weighted_similarities.size > 0 else 0.0
            candidate.score_topk_types_embedding = round(score, self.round_to_decimal_places)
