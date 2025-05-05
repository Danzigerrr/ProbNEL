from typing import List


class Candidate:
    score_types_embeddings_similarity = 0.0
    score_levenshtein_distance = 0.0
    score_popularity = 0.0
    score_context = 0.0
    score_final = 0.0

    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 ref_count: int,
                 dbpedia_score: float
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.ref_count = ref_count
        self.dbpedia_score = dbpedia_score

    def print_details(self):
        """Prints the details of the Candidate object."""
        print("- Candidate: -")
        print(f"Label: {self.label}")
        print(f"Ontology Types: {self.ontology_types}")
        print(f"Comment: {self.comment}")
        print(f"URI: {self.uri}")
        # print(f"Ref count (popularity): {self.ref_count}")
        print(f"Score NER to Ontology: {self.score_types_embeddings_similarity}")
        print(f"Score Levenshtein: {self.score_levenshtein_distance}")
        print(f"Score Popularity: {self.score_popularity}")
        print(f"Score Context: {self.score_context}")
        print(f"Score final: {self.score_final}")
