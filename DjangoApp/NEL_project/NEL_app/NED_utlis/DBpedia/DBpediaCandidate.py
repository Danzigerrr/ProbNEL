from typing import List, Optional


class Candidate:
    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 ref_count: int,
                 score_types_embeddings_similarity: Optional[float],
                 score_levenshtein_distance: Optional[float],
                 score_popularity: Optional[float],
                 score_final: Optional[float],
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.ref_count = ref_count
        self.score_types_embeddings_similarity = score_types_embeddings_similarity
        self.score_levenshtein_distance = score_levenshtein_distance
        self.score_popularity = score_popularity
        self.score_final = score_final

    def print_details(self):
        """Prints the details of the Candidate object."""
        print("- Candidate: -")
        print(f"Label: {self.label}")
        print(f"Ontology Types: {self.ontology_types}")
        print(f"Comment: {self.comment}")
        print(f"URI: {self.uri}")
        print(f"Ref count (popularity): {self.ref_count}")
        print(f"Score NER to Ontology: {self.score_types_embeddings_similarity}")
        print(f"Score Levenshtein: {self.score_levenshtein_distance}")
        print(f"Score Popularity: {self.score_popularity}")
        print(f"Score final: {self.score_final}")
