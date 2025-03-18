class TestEntity:
    def __init__(self, entity_label: str,
                 entity_type: str,
                 start_position: int,
                 end_position: int,
                 best_candidate_uri: str):
        self.entity_label = entity_label
        self.entity_type = entity_type
        self.start_position = start_position
        self.end_position = end_position
        self.best_candidate_uri = best_candidate_uri
