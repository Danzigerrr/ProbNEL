
class Text:
    def __init__(self, content: str):
        self.content = content
        self.entities = []  # List of associated Entity objects

    def add_entity(self, entity):
        self.entities.append(entity)
