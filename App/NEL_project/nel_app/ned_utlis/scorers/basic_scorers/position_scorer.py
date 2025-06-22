import pandas as pd
from App.NEL_project.nel_app.models.entity import Entity

class DbpediaCandidatePositionScorer:
    """
    scorers candidates based on their position in the candidate list (1 = best).
    """
    def __init__(self, round_to_decimal_places=3):
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, entity: Entity):
        if not entity.candidates:
            return

        positions = list(range(1, len(entity.candidates) + 1))
        df = pd.DataFrame({'score_position': positions})

        min_pos = df['score_position'].min()
        max_pos = df['score_position'].max()

        if max_pos == min_pos:
            normalized_scores = [1.0] * len(positions)
        else:
            normalized_scores = ((max_pos - df['score_position']) / (max_pos - min_pos)).round(self.round_to_decimal_places).tolist()

        for candidate, score in zip(entity.candidates, normalized_scores):
            candidate.score_position = score
