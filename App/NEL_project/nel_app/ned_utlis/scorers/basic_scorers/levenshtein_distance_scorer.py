from fuzzywuzzy import fuzz
from App.NEL_project.nel_app.models.entity import Entity

class LevenshteinDistanceScorer:
    """
    Calculates string similarity between entity label and candidate label using Levenshtein ratio.
    """
    def __init__(self, round_to_decimal_places=3):
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, entity: Entity):
        if not entity.candidates:
            return

        for candidate in entity.candidates:
            score = fuzz.ratio(entity.entity_label, candidate.label) / 100.0
            candidate.score_levenshtein = round(score, self.round_to_decimal_places)
