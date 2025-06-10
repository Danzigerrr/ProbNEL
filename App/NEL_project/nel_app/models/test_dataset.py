from App.NEL_project.nel_app.models.text import Text
from typing import List

class TestDataset:
    def __init__(self, name: str):
        self.texts: List[Text] = []
        self.name = name