from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from DjangoApp.NEL_project.NEL_app.Models.Text import Text
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity


class ContextScorer:
    """
    Class to calculate context similarity scores for candidates.
    """
    def __init__(self, round_to_decimal_places=3):
        self.vectorizer = TfidfVectorizer(max_features=1000)  # Limit features for memory efficiency
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, text: Text, entity: Entity):
        """
        Calculates context similarity scores for candidates using TF-IDF and cosine similarity.
        """
        if not entity.candidates:
            return

        entity_context = text.content  # Assuming text.content contains the relevant context
        for candidate in entity.candidates:
            candidate_context = candidate.comment

            # Vectorize the texts
            tfidf_matrix = self.vectorizer.fit_transform([entity_context, candidate_context])

            # Calculate cosine similarity
            similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

            candidate.score_context = round(similarity_score, self.round_to_decimal_places)
