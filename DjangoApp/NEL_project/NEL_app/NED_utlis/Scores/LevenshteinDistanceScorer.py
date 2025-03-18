from fuzzywuzzy import fuzz
from DjangoApp.NEL_project.NEL_app.classes import Entity


class LevenshteinDistanceScorer:
    """
    Class to calculate Levenshtein distance scores for candidates.
    """

    def __init__(self, round_to_decimal_places=3):
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, entity: Entity):
        """
        Calculates Levenshtein distance scores for candidates based on entity label and candidate label.
        """
        if not entity.candidates:
            return

        for candidate in entity.candidates:
            entity_label = entity.entity_label
            candidate_label = candidate.label

            # Calculate Levenshtein distance score using FuzzyWuzzy
            score_levenshtein_distance = fuzz.ratio(entity_label, candidate_label) / 100.0  # Normalize to 0-1 range

            candidate.score_levenshtein_distance = round(score_levenshtein_distance, self.round_to_decimal_places)
