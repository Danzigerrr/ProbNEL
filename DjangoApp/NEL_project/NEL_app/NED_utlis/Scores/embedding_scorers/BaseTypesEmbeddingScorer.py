import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class BaseTypesEmbeddingScorer:
    """
    Base class to calculate scores for candidates based on NER entity type and candidate ontology types.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", round_to_decimal_places=3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings_model = SentenceTransformer(model_name).to(self.device)
        self.round_to_decimal_places = round_to_decimal_places

    def get_embedding(self, sentence):
        """Generate word embedding using embeddings model."""
        return self.embeddings_model.encode(sentence, convert_to_tensor=True).to(self.device)

    def _calculate_similarity_matrix(self, ner_classes, ontology_types, ner_config):
        """Helper function to calculate the similarity matrix."""
        ner_embeddings_cache = {
            ner_type: self.get_embedding(ner_config.classes_definitions[ner_type])
            for ner_type, prob in ner_config.classes_definitions.items() if ner_type in [item[0] for item in ner_config.classes_definitions.items()] and ner_type != "O"
        }
        kg_embeddings = {kg_type: self.get_embedding(kg_type) for kg_type in ontology_types}
        similarity_matrix = np.zeros((len(ner_classes), len(ontology_types)))

        for i, ner_cls in enumerate(ner_classes):
            ner_embedding = ner_embeddings_cache.get(ner_cls)
            if ner_embedding is not None:
                for j, kg_type in enumerate(ontology_types):
                    kg_embedding = kg_embeddings.get(kg_type)
                    if kg_embedding is not None:
                        similarity_matrix[i, j] = util.pytorch_cos_sim(ner_embedding.unsqueeze(0), kg_embedding.unsqueeze(0)).item()
        return similarity_matrix
