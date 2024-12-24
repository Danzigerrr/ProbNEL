from typing import List


class TestDataset:
    def __init__(self, name):
        self.texts = []
        self.name = name

    def add_text(self, text):
        self.texts.append(text)


class Text:
    def __init__(self, content):
        self.content = content
        self.entities = []  # List of associated Entity objects

    def add_entity(self, entity):
        self.entities.append(entity)


class Entity:
    def __init__(self, entity_label: str, entity_type: str, start_position: int, end_position: int, dbpedia_uri: str, wikidata_uri: str, probabilities: List):
        self.entity_label = entity_label
        self.entity_type = entity_type
        self.start_position = start_position
        self.end_position = end_position
        self.dbpedia_uri = dbpedia_uri
        self.wikidata_uri = wikidata_uri
        self.probabilities = probabilities if probabilities is not None else []

