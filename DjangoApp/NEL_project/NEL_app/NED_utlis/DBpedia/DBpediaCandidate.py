from typing import List, Optional


class Candidate:
    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 score_types_embeddings_similarity: Optional[float],
                 score_levenshtein_distance: Optional[float],
                 score_final: Optional[float],
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.score_types_embeddings_similarity = score_types_embeddings_similarity
        self.score_levenshtein_distance = score_levenshtein_distance
        self.score_final = score_final

    def print_details(self):
        """Prints the details of the Candidate object."""
        print("- Candidate: -")
        print(f"Label: {self.label}")
        print(f"Ontology Types: {self.ontology_types}")
        print(f"Comment: {self.comment}")
        print(f"URI: {self.uri}")
        print(f"Score NER to Ontology: {self.score_types_embeddings_similarity}")
        print(f"Score Levenshtein: {self.score_levenshtein_distance}")
        print(f"Score final: {self.score_final}")
