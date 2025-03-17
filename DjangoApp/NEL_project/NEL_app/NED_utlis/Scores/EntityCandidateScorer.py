from typing import List
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.TypesEmbeddingScorer import TypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.classes import Entity
from DjangoApp.NEL_project.NEL_app.NED_utlis.DBpedia.DBpediaCandidate import Candidate


class EntityCandidateScorer:
    """
    Class to calculate scores for candidates based on NER entity type and candidate ontology types.
    """

    def __init__(self):
        self.typesEmbeddingScorer = TypesEmbeddingScorer()


    def calculate_scores_for_candidates(self, entity: Entity):
        """
        Calculates scores for candidates of an entity based on NER entity type and candidate ontology types.
        """
        self.typesEmbeddingScorer.calculate_score_types_embeddings_similarity(entity)

        self.normalise_scores(entity)

        self.calculate_final_score(entity)

    def normalise_scores(self, entity):
        normalize_scores(entity.candidates, "score_types_embeddings_similarity")

    def calculate_final_score(self, entity):
        for candidate in entity.candidates:
            candidate.score_final += candidate.score_types_embeddings_similarity


def normalize_scores(candidates: List[Candidate], score_attribute: str):
    """
    Normalizes candidate scores using min-max scaling for a specified score attribute.

    Args:
        candidates: A list of Candidate objects.
        score_attribute: The name of the score attribute to normalize (e.g., "score_types_embeddings_similarity", "score_final").
    """
    if not candidates:
        return

    try:
        scores = [getattr(candidate, score_attribute) for candidate in candidates]
    except AttributeError:
        print(f"Error: Score attribute '{score_attribute}' not found in Candidate objects.")
        return

    min_score = min(scores)
    max_score = max(scores)

    if max_score - min_score == 0:
        # All scores are the same, avoid division by zero
        for candidate in candidates:
            setattr(candidate, score_attribute, 1.0 if getattr(candidate, score_attribute) == max_score else 0.0)
        return

    for candidate in candidates:
        normalized_score = (getattr(candidate, score_attribute) - min_score) / (max_score - min_score)
        setattr(candidate, score_attribute, normalized_score)
