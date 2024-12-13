import json


class EvaluationResults:
    def __init__(self):
        # Initialize all variables to track evaluation metrics
        self.ned_accuracy = 0
        self.ner_accuracy = 0
        self.total_ner_ground_truth_entities = 0
        self.correct_ner_prediction_entities = 0
        self.total_ned_ground_truth_uris = 0
        self.correct_ned_uris_prediction = 0

    def update_metrics(self, ground_truth_entity, matching_entity):
        """Update metrics based on a ground-truth entity and its matching predicted entity."""
        self.total_ner_ground_truth_entities += 1

        if ground_truth_entity.get("uri"):
            self.total_ned_ground_truth_uris += 1

        if matching_entity:
            self.correct_ner_prediction_entities += 1

            if ground_truth_entity.get("uri") and matching_entity.get("uri") == ground_truth_entity["uri"]:
                self.correct_ned_uris_prediction += 1

    def calculate_ner_accuracy(self):
        """Calculate NER accuracy."""
        return (
                       self.correct_ner_prediction_entities / self.total_ner_ground_truth_entities) * 100 if self.total_ner_ground_truth_entities > 0 else 0

    def calculate_ned_accuracy(self):
        """Calculate NED accuracy."""
        return (
                       self.correct_ned_uris_prediction / self.total_ned_ground_truth_uris) * 100 if self.total_ned_ground_truth_uris > 0 else 0

    def finalize(self):
        """Finalize the evaluation by calculating the accuracies."""
        self.ner_accuracy = self.calculate_ner_accuracy()
        self.ned_accuracy = self.calculate_ned_accuracy()

    def print(self):
        """Print the evaluation results."""
        print("NER and NED Evaluation:")
        print(f"Total ground-truth mentions: {self.total_ner_ground_truth_entities}")
        print(f"Correctly predicted mentions (NER): {self.correct_ner_prediction_entities}")
        print(f"NER Accuracy: {self.calculate_ner_accuracy():.2f}%")
        print(f"Total ground-truth URIs: {self.total_ned_ground_truth_uris}")
        print(f"Correctly matched URIs (NED): {self.correct_ned_uris_prediction}")
        print(f"NED Accuracy: {self.calculate_ned_accuracy():.2f}%")

    def to_json(self):
        """Convert the evaluation results into a JSON-serializable dictionary."""
        return json.dumps({
            "total_ner_ground_truth_entities": self.total_ner_ground_truth_entities,
            "correct_ner_prediction_entities": self.correct_ner_prediction_entities,
            "ner_accuracy": f"{self.calculate_ner_accuracy():.2f}%",
            "total_ned_ground_truth_uris": self.total_ned_ground_truth_uris,
            "correct_ned_uris_prediction": self.correct_ned_uris_prediction,
            "ned_accuracy": f"{self.calculate_ned_accuracy():.2f}%",
        })
