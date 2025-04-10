from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.TypesEmbeddingScorer import TypesEmbeddingScorer
from DjangoApp.NEL_project.NEL_app.NED_utlis.Scores.ContextScorer import ContextScorer
from DjangoApp.NEL_project.NEL_app.model_components.Text import Text
from DjangoApp.NEL_project.NEL_app.model_components.Entity import Entity
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig


class EntityCandidateScorer:
    """
    Class to calculate scores for candidates of each entity in text.
    """

    def __init__(self):
        self.typesEmbeddingScorer = TypesEmbeddingScorer()
        self.ContextScorer = ContextScorer()

    def calculate_scores_for_candidates(self, text: Text, entity: Entity, ner_config: NERConfig, use_score_types_embeddings_similarity = True):
        """
        Calculates scores for candidates of an entity from a text.
        """
        if use_score_types_embeddings_similarity:
            self.typesEmbeddingScorer.calculate_score(entity, ner_config)

        self.ContextScorer.calculate_score(text, entity)

        self.calculate_final_score(entity)

    def calculate_final_score(self, entity: Entity):
        if entity.candidates:
            for candidate in entity.candidates:
                candidate.score_final += candidate.score_types_embeddings_similarity
                candidate.score_final += candidate.score_context
                candidate.score_final = round(number=candidate.score_final, ndigits=3)
