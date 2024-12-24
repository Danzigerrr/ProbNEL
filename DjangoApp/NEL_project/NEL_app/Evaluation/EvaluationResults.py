import json


class EvaluationResults:
    def __init__(self):
        # Initialize NER-specific metrics
        self.total_ground_truth_entities = 0
        self.correct_predicted_entities = 0
        self.accuracy_score = 0

    def update_metrics(self, matching_entity):
        """Update NER metrics based on a ground-truth entity and its matching predicted entity."""
        self.total_ground_truth_entities += 1
        if matching_entity:
            self.correct_predicted_entities += 1

    def finalize(self):
        """Finalize the NER evaluation by calculating the accuracy."""
        self.calculate_accuracy()

    def calculate_accuracy(self):
        """Calculate NER accuracy."""
        if self.total_ground_truth_entities > 0:
            self.accuracy_score = (self.correct_predicted_entities / self.total_ground_truth_entities) * 100
        else:
            self.accuracy_score = 0

    def to_json(self):
        """Convert the evaluation results into a JSON-serializable dictionary."""
        return json.dumps({
            "total_ground_truth_entities": self.total_ground_truth_entities,
            "correct_predicted_entities": self.correct_predicted_entities,
            "accuracy_score": f"{self.accuracy_score:.2f}%"
        })
