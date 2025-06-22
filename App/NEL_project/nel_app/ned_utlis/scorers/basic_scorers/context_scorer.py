from sentence_transformers import SentenceTransformer, util
from App.NEL_project.nel_app.models.entity import Entity

class ContextScorer:
    """
    Calculates similarity between text and candidate comments using a bi-encoder SBERT model.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", round_to_decimal_places=3):
        self.model = SentenceTransformer(model_name)
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, text, entity: Entity):
        if not entity.candidates:
            return

        entity_embedding = self.model.encode(text.content, convert_to_tensor=True)

        for candidate in entity.candidates:
            candidate_embedding = self.model.encode(candidate.comment, convert_to_tensor=True)
            score = util.pytorch_cos_sim(entity_embedding, candidate_embedding).item()
            candidate.score_context = round(score, self.round_to_decimal_places)
