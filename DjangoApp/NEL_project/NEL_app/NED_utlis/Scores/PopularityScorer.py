import numpy as np
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity

class PopularityScorer:
    """
    Calculates popularity score using log(ref_count) normalization across all candidates.
    """
    def __init__(self, round_to_decimal_places=3):
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, entity: Entity):
        if not entity.candidates:
            return

        ref_counts = [candidate.ref_count for candidate in entity.candidates]
        log_counts = np.log1p(ref_counts)

        min_log = np.min(log_counts) if len(log_counts) > 0 else 0
        max_log = np.max(log_counts) if len(log_counts) > 0 else 1

        if max_log - min_log == 0:
            for candidate in entity.candidates:
                candidate.score_popularity = 1.0
        else:
            for i, candidate in enumerate(entity.candidates):
                normalized = (log_counts[i] - min_log) / (max_log - min_log)
                candidate.score_popularity = round(normalized, self.round_to_decimal_places)
