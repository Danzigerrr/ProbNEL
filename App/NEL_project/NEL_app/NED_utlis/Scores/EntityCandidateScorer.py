from App.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.TypesEmbeddingScorer import TypesEmbeddingScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.TopKTypesEmbeddingScorer import TopKTypesEmbeddingScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.embedding_scorers.MaxNERTypesEmbeddingScorer import MaxNERTypesEmbeddingScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.basic_scorers.LevenshteinDistanceScorer import LevenshteinDistanceScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.basic_scorers.PopularityScorer import PopularityScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.basic_scorers.ContextScorer import ContextScorer
from App.NEL_project.NEL_app.NED_utlis.Scores.basic_scorers.PositionScorer import DbpediaCandidatePositionScorer
from App.NEL_project.NEL_app.Models.Text import Text
from App.NEL_project.NEL_app.Models.Entity import Entity
from App.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


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
