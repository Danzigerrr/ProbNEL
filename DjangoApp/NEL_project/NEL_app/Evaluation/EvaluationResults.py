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

    def update_metrics(self, ground_truth_exists, predicted_exists, matched):
        """Update evaluation metrics based on ground truth and predicted entities."""
        if ground_truth_exists:
            self.total_ground_truth_entities += 1
        if predicted_exists and matched:
            self.correct_predicted_entities += 1

    def calculate_scores(self, predicted_entities, ground_truth_entities):
        """Calculates precision, recall, F1 score and accuracy."""
        y_true = []
        y_pred = []

        for gt_entity in ground_truth_entities:
            y_true.append(1)  # Ground truth entity exists
            found_match = False
            for pred_entity in predicted_entities:
                if (gt_entity.start_position == pred_entity.start_position and
                        gt_entity.end_position == pred_entity.end_position and
                        gt_entity.best_candidate_uri == pred_entity.best_candidate_uri):
                    y_pred.append(1)  # Predicted entity matches ground truth
                    found_match = True
                    break
            if not found_match:
                y_pred.append(0)  # Predicted entity does not match

        # Handle cases where predicted entities exist but no ground truth
        for pred_entity in predicted_entities:
            gt_match = False
            for gt_entity in ground_truth_entities:
                if (gt_entity.start_position == pred_entity.start_position and
                        gt_entity.end_position == pred_entity.end_position):
                    gt_match = True
                    break
            if not gt_match:
                y_true.append(0)
                y_pred.append(1)

        if len(y_true) > 0:
            self.precision = precision_score(y_true, y_pred, zero_division=0)
            self.recall = recall_score(y_true, y_pred, zero_division=0)
            self.f1_score = f1_score(y_true, y_pred, zero_division=0)
            self.accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)

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
