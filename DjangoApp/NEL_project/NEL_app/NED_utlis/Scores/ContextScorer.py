from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import euclidean

from DjangoApp.NEL_project.NEL_app.model_components.Entity import Entity


class ContextScorer:
    """
    Class to calculate context similarity scores for candidates using SBERT.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", round_to_decimal_places=3):
        """
        Initializes the ContextScorer with an SBERT model.

        :param model_name: Name of the SBERT model to be used.
        :param round_to_decimal_places: Number of decimal places to round the similarity scores.
        """
        self.model = SentenceTransformer(model_name)
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, text, entity: Entity):
        """
        Calculates context similarity scores for candidates using SBERT embeddings,
        both cosine similarity and Euclidean distance.

        :param text: The text containing the named entity.
        :param entity: The entity for which candidate similarity scores are calculated.
        """
        if not entity.candidates:
            return

        entity_context = text.content  # Assuming text.content contains the relevant context
        entity_embedding = self.model.encode(entity_context, convert_to_tensor=True)

        for candidate in entity.candidates:
            candidate_context = candidate.comment
            candidate_embedding = self.model.encode(candidate_context, convert_to_tensor=True)

            # Compute Cosine Similarity
            cosine_similarity_score = util.pytorch_cos_sim(entity_embedding, candidate_embedding).item()
            candidate.score_context_cosine = round(cosine_similarity_score, self.round_to_decimal_places)

            # Compute Euclidean Distance (converted to a similarity score)
            euclidean_distance_score = euclidean(entity_embedding.cpu(), candidate_embedding.cpu())
            euclidean_similarity_score = 1 / (1 + euclidean_distance_score)  # Inverse to make it a similarity score
            candidate.score_context_euclidean = round(euclidean_similarity_score, self.round_to_decimal_places)
