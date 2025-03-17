from typing import List, Optional


class Candidate:
    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 score_ner_to_ontology: Optional[float],
                 candidate_score: Optional[float],
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.score_ner_to_ontology = score_ner_to_ontology
        self.candidate_score = candidate_score

    def print_details(self):
        """Prints the details of the Candidate object."""
        print("- Candidate: -")
        print(f"Label: {self.label}")
        print(f"Ontology Types: {self.ontology_types}")
        print(f"Comment: {self.comment}")
        print(f"URI: {self.uri}")
        print(f"Score NER to Ontology: {self.score_ner_to_ontology}")
        print(f"Score final: {self.candidate_score}")
