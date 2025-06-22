from App.NEL_project.nel_app.ned_utlis.scorers.embedding_scorers.types_embedding_scorer import TypesEmbeddingScorer
from App.NEL_project.nel_app.ned_utlis.scorers.embedding_scorers.top_k_types_embedding_scorer import TopKTypesEmbeddingScorer
from App.NEL_project.nel_app.ned_utlis.scorers.embedding_scorers.max_ner_types_embedding_scorer import MaxNERTypesEmbeddingScorer
from App.NEL_project.nel_app.ned_utlis.scorers.basic_scorers.levenshtein_distance_scorer import LevenshteinDistanceScorer
from App.NEL_project.nel_app.ned_utlis.scorers.basic_scorers.popularity_scorer import PopularityScorer
from App.NEL_project.nel_app.ned_utlis.scorers.basic_scorers.context_scorer import ContextScorer
from App.NEL_project.nel_app.ned_utlis.scorers.basic_scorers.position_scorer import DbpediaCandidatePositionScorer
from App.NEL_project.nel_app.models.text import Text
from App.NEL_project.nel_app.models.entity import Entity
from App.NEL_project.nel_app.ner_utils.ner_config import NERConfig


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
            self.typesBasicEmbeddingScorer = TypesEmbeddingScorer()
            self.typesTopkEmbeddingScorer = TopKTypesEmbeddingScorer()
            self.typesMaxnerEmbeddingScorer = MaxNERTypesEmbeddingScorer()

    def calculate_scores_for_candidates(self, text: Text, entity: Entity, ner_config: NERConfig, use_score_types_embeddings_similarity = True):
        """
        Calculates scores for candidates of an entity from a text.
        """
        if use_score_types_embeddings_similarity:
            self.typesBasicEmbeddingScorer.calculate_score(entity, ner_config)
            self.typesTopkEmbeddingScorer.calculate_score(entity, ner_config)
            self.typesMaxnerEmbeddingScorer.calculate_score(entity, ner_config)

        self.LevenshteinDistanceScorer.calculate_score(entity)
        self.PopularityScorer.calculate_score(entity)
        self.ContextScorer.calculate_score(text, entity)
        self.PositionScorer.calculate_score(entity)
