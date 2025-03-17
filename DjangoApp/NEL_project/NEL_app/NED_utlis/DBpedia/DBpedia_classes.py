from typing import List, Optional


class Candidate:
    def __init__(self,
                 label: str,
                 ontology_types: List[str],
                 comment: str,
                 uri: str,
                 score_ner_to_ontology: Optional[float],
                 ):
        self.label = label
        self.ontology_types = ontology_types
        self.comment = comment
        self.uri = uri
        self.score_ner_to_ontology = score_ner_to_ontology
