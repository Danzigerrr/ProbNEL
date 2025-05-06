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
