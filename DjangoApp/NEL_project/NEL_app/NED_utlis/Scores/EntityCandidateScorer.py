from typing import List
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.TypesEmbeddingScorer import TypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.LevenshteinDistanceScorer import LevenshteinDistanceScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.PopularityScorer import PopularityScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.ContextScorer import ContextScorer
from DjangoApp.NEL_project.NEL_app.Models.Text import Text
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.NED_utlis.Candidate.Candidate import Candidate
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


class EntityCandidateScorer:
    """
    Class to calculate scores for candidates of each entity in text.
    """

    def __init__(self):
        self.typesEmbeddingScorer = TypesEmbeddingScorer()
        self.types_embedding_score_factor = 1

        self.LevenshteinDistanceScorer = LevenshteinDistanceScorer()
        self.levenshtein_distance_score_factor = 1

        self.PopularityScorer = PopularityScorer()
        self.popularity_score_factor = 1

        self.ContextScorer = ContextScorer()
        self.context_score_factor = 1

    def calculate_scores_for_candidates(self, text: Text, entity: Entity, ner_config: NERConfig):
        """
        Calculates scores for candidates of an entity from a text.
        """
        self.typesEmbeddingScorer.calculate_score(entity, ner_config)
        self.LevenshteinDistanceScorer.calculate_score(entity)
        self.PopularityScorer.calculate_score(entity)
        self.ContextScorer.calculate_score(text, entity)

        self.calculate_final_score(entity)

    def calculate_final_score(self, entity):
        if entity.candidates:
            for candidate in entity.candidates:
                candidate.score_final += self.types_embedding_score_factor * candidate.score_types_embeddings_similarity
                candidate.score_final += self.levenshtein_distance_score_factor * candidate.score_levenshtein_distance
                candidate.score_final += self.popularity_score_factor * candidate.score_popularity
                candidate.score_final += self.context_score_factor * candidate.score_context
                candidate.score_final = round(number=candidate.score_final, ndigits=3)


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
