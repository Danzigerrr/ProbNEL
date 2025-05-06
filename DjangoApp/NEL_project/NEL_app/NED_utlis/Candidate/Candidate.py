from typing import List


class Candidate:
    score_types_embeddings_similarity = 0.0
    score_levenshtein = 0.0
    score_popularity = 0.0
    score_context = 0.0
    score_position = 0.0
    score_basic_types_embedding = 0.0
    score_topk_types_embedding = 0.0
    score_maxner_types_embedding = 0.0

    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 ref_count: int,
                 position: int
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.ref_count = ref_count
        self.position = position

    def print_details(self):
        """Prints the details of the Candidate object."""
        print("- Candidate: -")
        print(f"Label: {self.label}")
        print(f"Ontology Types: {self.ontology_types}")
        print(f"Comment: {self.comment}")
        print(f"URI: {self.uri}")

