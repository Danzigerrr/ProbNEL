from ..Models.Text import Text


class TestDataset:
    def __init__(self, name: str):
        self.texts = []
        self.name = name

    def add_text(self, text: Text):
        self.texts.append(text)