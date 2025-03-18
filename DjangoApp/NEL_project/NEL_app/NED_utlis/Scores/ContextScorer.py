from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from DjangoApp.NEL_project.NEL_app.classes import Text, Entity


class ContextScorer:
    """
    Class to calculate context similarity scores for candidates.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)  # Limit features for memory efficiency

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

            candidate.score_context = similarity_score
