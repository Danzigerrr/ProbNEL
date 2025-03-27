from DjangoApp.NEL_project.NEL_app.Models.Text import Text


class TestText(Text):
    def __init__(self, content: str):
        super().__init__(content)
        self.logs = []
