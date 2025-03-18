class TestEntity:
    def __init__(self, entity_label: str,
                 start_position: int,
                 end_position: int,
                 target_uri: str):
        self.entity_label = entity_label
        self.start_position = start_position
        self.end_position = end_position
        self.target_uri = target_uri
