import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class EvaluationResults:
    def __init__(self):
        # Initialize evaluation metrics and lists for storing true and predicted labels
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.accuracy = 0.0
        self.y_true = []  # List to store ground truth labels (1 for entity, 0 for non-entity)
        self.y_pred = []  # List to store predicted labels

    def evaluate_ned_process(self, predicted_entities, ground_truth_entities):
        """Calculates precision, recall, F1 score, and accuracy."""
        self.evaluate_ground_truth_entities(predicted_entities, ground_truth_entities)
        self.evaluate_predicted_entities(predicted_entities, ground_truth_entities)

    def evaluate_ground_truth_entities(self, predicted_entities, ground_truth_entities):
        """Iterate through ground truth entities and evaluate if they are correctly predicted."""

        for gt_entity in ground_truth_entities:
            found_match = False

            for pred_entity in predicted_entities:
                # Check if predicted entity matches a ground truth entity by position and target URI
                if check_ned_matching(gt_entity, pred_entity):
                    found_match = True
                    break  # Stop searching once a correct match is found

            # Store evaluation labels (1 if entity exists, 0 otherwise)
            self.y_true.append(1)  # Ground truth entity exists
            self.y_pred.append(1 if found_match else 0)  # Correct prediction (1) or incorrect prediction (0)

    def evaluate_predicted_entities(self, predicted_entities, ground_truth_entities):
        """Iterate through predicted entities to identify false positives."""
        for pred_entity in predicted_entities:
            gt_match = False

            for gt_entity in ground_truth_entities:
                if check_ner_matching(gt_entity, pred_entity):
                    gt_match = True
                    break  # Stop searching once a matching ground truth entity is found

            if not gt_match:
                self.y_true.append(0)  # No ground truth entity
                self.y_pred.append(1)  # Incorrectly predicted entity


    def finalize_scores(self):
        """Finalizes and calculates the evaluation metrics."""
        if len(self.y_true) > 0:
            # Compute precision, recall, and F1-score, handling zero-division cases
            self.precision = precision_score(self.y_true, self.y_pred, zero_division=0)
            self.recall = recall_score(self.y_true, self.y_pred, zero_division=0)
            self.f1_score = f1_score(self.y_true, self.y_pred, zero_division=0)
            # Compute accuracy as correctly classified instances over total instances
            self.accuracy = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == pred) / len(self.y_true)

            # Display the confusion matrix
            self.show_confusion_matrix()


    def show_confusion_matrix(self):
        """Displays the confusion matrix for predictions."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"Confusion Matrix for NED Predictions:\n{cm}")


    def to_json(self):
        """Convert the evaluation results into a JSON-serializable dictionary."""
        return json.dumps({
            "accuracy": f"{self.accuracy:.4f}",
            "precision": f"{self.precision:.4f}",
            "recall": f"{self.recall:.4f}",
            "f1_score": f"{self.f1_score:.4f}"
        })

    def print_results(self):
        """Prints the evaluation results to the console in a formatted way."""
        print("Evaluation Results:")
        print("--------------------")
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1_score:.4f}")

def check_ned_matching(gt_entity, pred_entity):
    """Checks if a predicted entity matches a ground truth entity based on NED criteria."""
    return (
            check_ner_matching(gt_entity, pred_entity) and
            gt_entity.target_uri == pred_entity.best_candidate_uri
    )

def check_ner_matching(gt_entity, pred_entity):
    """Checks if a predicted entity matches a ground truth entity based on NER criteria."""
    return (
            gt_entity.start_position == pred_entity.start_position and
            gt_entity.end_position == pred_entity.end_position
    )