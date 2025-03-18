import json
from sklearn.metrics import precision_score, recall_score, f1_score


class EvaluationResults:
    def __init__(self):
        self.total_ground_truth_entities = 0
        self.correct_predicted_entities = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.accuracy = 0.0
        self.y_true = []
        self.y_pred = []

    def calculate_scores(self, predicted_entities, ground_truth_entities):
        """Calculates precision, recall, F1 score and accuracy."""

        # Count total ground truth entities
        self.total_ground_truth_entities += len(ground_truth_entities)

        # Iterate through ground truth entities and check for matches
        for gt_entity in ground_truth_entities:
            found_match = False
            for pred_entity in predicted_entities:
                if (gt_entity.start_position == pred_entity.start_position and
                        gt_entity.end_position == pred_entity.end_position and
                        gt_entity.target_uri == pred_entity.best_candidate_uri):
                    found_match = True
                    self.correct_predicted_entities += 1
                    break

            self.y_true.append(1)  # Ground truth entity exists
            self.y_pred.append(1 if found_match else 0)  # Predicted entity matches or not

        # Handle cases where predicted entities exist but no ground truth
        for pred_entity in predicted_entities:
            gt_match = False
            for gt_entity in ground_truth_entities:
                if (gt_entity.start_position == pred_entity.start_position and
                        gt_entity.end_position == pred_entity.end_position):
                    gt_match = True
                    break
            if not gt_match:
                self.y_true.append(0)
                self.y_pred.append(1)

    def finalize_scores(self):
        """Finalizes the scores after all texts have been processed."""
        if len(self.y_true) > 0:
            self.precision = precision_score(self.y_true, self.y_pred, zero_division=0)
            self.recall = recall_score(self.y_true, self.y_pred, zero_division=0)
            self.f1_score = f1_score(self.y_true, self.y_pred, zero_division=0)
            self.accuracy = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == pred) / len(self.y_true)

    def to_json(self):
        """Convert the evaluation results into a JSON-serializable dictionary."""
        return json.dumps({
            "total_ground_truth_entities": self.total_ground_truth_entities,
            "correct_predicted_entities": self.correct_predicted_entities,
            "precision": f"{self.precision:.4f}",
            "recall": f"{self.recall:.4f}",
            "f1_score": f"{self.f1_score:.4f}",
            "accuracy": f"{self.accuracy:.4f}"
        })
