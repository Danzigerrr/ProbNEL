import numpy as np
from DjangoApp.NEL_project.NEL_app.classes import Entity


class PopularityScorer:
    """
    Class to calculate popularity scores for candidates based on refCount.
    """
    def __init__(self, round_to_decimal_places=3):
        self.round_to_decimal_places = round_to_decimal_places

    def calculate_score(self, entity: Entity):
        """
        Calculates popularity scores for candidates based on refCount, using log transformation and normalization.
        """
        if not entity.candidates:
            return

        ref_counts = [candidate.ref_count for candidate in entity.candidates]

        # Apply log transformation to handle large differences
        log_ref_counts = np.log1p(ref_counts)  # log(1 + x) to handle 0 values

        # Normalize log-transformed scores to a 0-1 range
        if len(log_ref_counts) > 0:
            min_log_ref_count = np.min(log_ref_counts)
            max_log_ref_count = np.max(log_ref_counts)

            if max_log_ref_count - min_log_ref_count == 0:
                # All values are same
                for candidate in entity.candidates:
                    candidate.score_popularity = 1.0 if candidate.ref_count == max(ref_counts) else 0.0

            else:
                for i, candidate in enumerate(entity.candidates):
                    normalized_score = (log_ref_counts[i] - min_log_ref_count) / (max_log_ref_count - min_log_ref_count)
                    candidate.score_popularity = round(normalized_score, self.round_to_decimal_places)
