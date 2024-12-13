class TestDataset:
    def __init__(self, name):
        self.texts = []
        self.name = name

    def add_text(self, text):
        self.texts.append(text)


class Text:
    def __init__(self, content):
        self.content = content
        self.entities = []  # List of associated FoundEntity objects

    def add_entity(self, entity):
        self.entities.append(entity)

    def __str__(self):
        return self.content[:50]  # Display first 50 characters


class FoundEntity:
    def __init__(self, text, entity_label, entity_type, start_position, end_position, uri=None, probabilities=None):
        self.text = text  # Reference to the parent Text object
        self.entity_label = entity_label
        self.entity_type = entity_type
        self.start_position = start_position
        self.end_position = end_position
        self.uri = uri
        self.probabilities = probabilities if probabilities is not None else []

    def __str__(self):
        return self.entity_label


class OriginalEntity:
    def __init__(self, surface_form, position_start, position_end, dbpedia_uri, wikidata_uri):
        self.surface_form = surface_form
        self.position_start = position_start
        self.position_end = position_end
        self.dbpedia_uri = dbpedia_uri
        self.wikidata_uri = wikidata_uri
