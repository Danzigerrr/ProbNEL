from typing import List


class Entity:
    def __init__(self, entity_label: str, entity_type: str, start_position: int, end_position: int,
                 best_candidate_uri: str, probabilities: List):
        self.entity_label = entity_label
        self.entity_type = entity_type
        self.start_position = start_position
        self.end_position = end_position
        self.best_candidate_uri = best_candidate_uri
        self.probabilities = probabilities if probabilities is not None else []
        self.candidates = []


class Text:
    def __init__(self, content: str):
        self.content = content
        self.entities = []  # List of associated Entity objects

    def add_entity(self, entity):
        self.entities.append(entity)


class TestDataset:
    def __init__(self, name: str):
        self.texts = []
        self.name = name

    def add_text(self, text: Text):
        self.texts.append(text)
