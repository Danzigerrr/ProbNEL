import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from scipy.spatial.distance import cosine

from DjangoApp.NEL_project.NEL_app.classes import Entity


class EntityCandidateScorer:
    """
    Class to calculate scores for candidates based on NER entity type and candidate ontology types.
    """

    def __init__(self):
        self.stacked_embeddings = StackedEmbeddings([
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward-fast'),
            FlairEmbeddings('news-backward-fast'),
        ])

    def get_embedding(self, word):
        """Generate word embedding using Flair."""
        sentence = Sentence(word)
        self.stacked_embeddings.embed(sentence)
        return sentence[0].embedding.cpu().detach().numpy()

    def calculate_candidate_scores(self, entity: Entity):
        """
        Calculates scores for candidates of an entity based on NER entity type and candidate ontology types.
        """
        if not entity.candidates:
            return

        ner_classes = [item[0] for item in entity.probabilities]
        ner_probabilities = np.array([float(item[1]) for item in entity.probabilities])

        for candidate in entity.candidates:
            kg_types = candidate.ontology_types

            ner_embeddings = {cls: self.get_embedding(cls) for cls in ner_classes}
            kg_embeddings = {kg_type: self.get_embedding(kg_type) for kg_type in kg_types}

            similarity_matrix = np.zeros((len(ner_classes), len(kg_types)))

            for i, ner_cls in enumerate(ner_classes):
                for j, kg_type in enumerate(kg_types):
                    similarity_matrix[i, j] = 1 - cosine(ner_embeddings[ner_cls], kg_embeddings[kg_type])

            final_scores = {}

            for j, kg_type in enumerate(kg_types):
                max_similarities = np.max(similarity_matrix[:, j])
                weighted_score = sum(ner_probabilities[i] * similarity_matrix[i, j] for i in range(len(ner_classes)))

                final_scores[kg_type] = weighted_score

            candidate_score = 0
            for ontology_type in candidate.ontology_types:
                max_similarity_per_ner_class = np.max(similarity_matrix[:, kg_types.index(ontology_type)])
                candidate_score += sum(ner_probabilities * similarity_matrix[:, kg_types.index(ontology_type)])
            candidate.candidate_score = candidate_score/len(candidate.ontology_types) if len(candidate.ontology_types) > 0 else 0


