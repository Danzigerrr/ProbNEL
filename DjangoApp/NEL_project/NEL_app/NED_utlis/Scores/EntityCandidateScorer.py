from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.TypesEmbeddingScorer import TypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.LevenshteinDistanceScorer import LevenshteinDistanceScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.PopularityScorer import PopularityScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.ContextScorer import ContextScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.PositionScorer import DbpediaCandidatePositionScorer
from DjangoApp.NEL_project.NEL_app.Models.Text import Text
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


class EntityCandidateScorer:
    """
    Class to calculate scores for candidates of each entity in text.
    """

    def __init__(self, use_score_types_embeddings_similarity: bool):
        self.LevenshteinDistanceScorer = LevenshteinDistanceScorer()
        self.PopularityScorer = PopularityScorer()
        self.ContextScorer = ContextScorer()
        self.PositionScorer = DbpediaCandidatePositionScorer()

        if use_score_types_embeddings_similarity:
            self.typesEmbeddingScorer = TypesEmbeddingScorer()

    def calculate_scores_for_candidates(self, text: Text, entity: Entity, ner_config: NERConfig, use_score_types_embeddings_similarity = True):
        """
        Calculates scores for candidates of an entity from a text.
        """
        if use_score_types_embeddings_similarity:
            self.typesEmbeddingScorer.calculate_score(entity, ner_config)

        self.LevenshteinDistanceScorer.calculate_score(entity)
        self.PopularityScorer.calculate_score(entity)
        self.ContextScorer.calculate_score(text, entity)
        self.PositionScorer.calculate_score(entity)

        self.calculate_final_score(entity)

    def calculate_final_score(self, entity):
        if entity.candidates:
            for candidate in entity.candidates:
                candidate.score_final += candidate.score_types_embeddings_similarity
                candidate.score_final += candidate.score_levenshtein_distance
                candidate.score_final += candidate.score_popularity
                candidate.score_final += candidate.score_context
                candidate.score_final += candidate.score_position
                candidate.score_final = round(number=candidate.score_final, ndigits=3)
