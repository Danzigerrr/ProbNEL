from ..Models.Text import Text
from typing import List

class TestDataset:
    def __init__(self, name: str):
        self.texts: List[Text] = []
        self.name = name