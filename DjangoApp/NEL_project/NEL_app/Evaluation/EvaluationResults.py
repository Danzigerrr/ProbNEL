import json
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from DjangoApp.NEL_project.NEL_app.Evaluation.TestText import TestText


# import matplotlib.pyplot as plt
# import seaborn as sns

class EvaluationResults:
    def __init__(self):
        # Initialize evaluation metrics and lists for storing true and predicted labels
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.accuracy = 0.0
        self.y_true = []  # List to store ground truth labels (1 for entity, 0 for non-entity)
        self.y_pred = []  # List to store predicted labels

    def calculate_scores(self, text_to_analyse: TestText, predicted_entities, ground_truth_entities):
        """Calculates precision, recall, F1 score, and accuracy."""
        self.evaluate_ground_truth_entities(text_to_analyse, predicted_entities, ground_truth_entities)
        self.evaluate_predicted_entities(predicted_entities, ground_truth_entities)

    def evaluate_ground_truth_entities(self, text_to_analyse, predicted_entities, ground_truth_entities):
        """Iterate through ground truth entities and evaluate if they are correctly predicted."""
        print("$$ Iterating through Ground Truth entities:")

        for gt_entity in ground_truth_entities:
            found_match = False
            matched_pred_entity = None

            for pred_entity in predicted_entities:
                # Check if predicted entity matches a ground truth entity by position and target URI
                if check_ned_matching(gt_entity, pred_entity):
                    found_match = True
                    matched_pred_entity = pred_entity
                    break  # Stop searching once a correct match is found

            if found_match:
                print(f"$ Correct prediction found for '{gt_entity.entity_label}'({gt_entity.start_position}, {gt_entity.end_position})")
                log_entry = {
                    "entity_label": gt_entity.entity_label,
                    "start_position": gt_entity.start_position,
                    "end_position": gt_entity.end_position,
                    "correct_prediction": True,
                    "matching_candidate_index": 0,
                    "candidates": [(c.label, c.score_types_embeddings_similarity, c.score_levenshtein_distance, c.score_popularity, c.score_context) for c in matched_pred_entity.candidates]
                }
                print(f"  Logs - Candidates: (Id:0) {log_entry['candidates']}")
                text_to_analyse.logs.append(log_entry)
                print(f"Logs updated ({len(text_to_analyse.logs)}): {text_to_analyse.logs}")
            else:
                print(f"$ No Correct prediction found for '{gt_entity.entity_label}'({gt_entity.start_position}, {gt_entity.end_position})")

                # Search for the matching entity in the candidate list and return its index
                entity_with_cand, candidate_index = self.find_candidate_index(gt_entity, predicted_entities)
                if entity_with_cand is not None:
                    log_entry = {
                        "entity_label": gt_entity.entity_label,
                        "start_position": gt_entity.start_position,
                        "end_position": gt_entity.end_position,
                        "correct_prediction": False,
                        "matching_candidate_index": candidate_index,
                        "candidates": [(c.label, c.score_types_embeddings_similarity, c.score_levenshtein_distance, c.score_popularity, c.score_context) for c in entity_with_cand.candidates]
                    }
                    print(f"  Logs - But one of the candidates is matching: Candidates: (Id:{candidate_index}) {log_entry['candidates']}")
                    text_to_analyse.logs.append(log_entry)
                    print(f"Logs updated ({len(text_to_analyse.logs)}): {text_to_analyse.logs}")

            # Store evaluation labels (1 if entity exists, 0 otherwise)
            self.y_true.append(1)  # Ground truth entity exists
            self.y_pred.append(1 if found_match else 0)  # Correct prediction (1) or incorrect prediction (0)

    def find_candidate_index(self, gt_entity, predicted_entities):
        """Search for the target_uri in all predicted entities' candidate lists and return its index.
        If not found, return None.
        """
        for pred_entity in predicted_entities:
            if check_ner_matching(gt_entity, pred_entity):
                for idx, candidate in enumerate(pred_entity.candidates):
                    if candidate.uri == gt_entity.target_uri:
                        return pred_entity, idx  # Return the index if the URIs match
        return None, None


    def evaluate_predicted_entities(self, predicted_entities, ground_truth_entities):
        """Iterate through predicted entities to identify false positives."""
        print("$$ Iterating through found entities:")

        for pred_entity in predicted_entities:
            gt_match = False

            for gt_entity in ground_truth_entities:
                if (gt_entity.start_position == pred_entity.start_position and
                        gt_entity.end_position == pred_entity.end_position):
                    gt_match = True
                    break  # Stop searching once a matching ground truth entity is found

            if not gt_match:
                self.y_true.append(0)  # No ground truth entity
                self.y_pred.append(1)  # Incorrectly predicted entity
                print(f"$ Predicted entity '{pred_entity.entity_label}'({pred_entity.start_position}, {pred_entity.end_position}) of type {pred_entity.probabilities[0]} does not exist in the ground truth text")


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
        print(f"Confusion Matrix:\n{cm}")
        # plt.figure(figsize=(5, 4))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Entity', 'Entity'], yticklabels=['No Entity', 'Entity'])
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')
        # plt.show()

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
            gt_entity.start_position == pred_entity.start_position and
            gt_entity.end_position == pred_entity.end_position and
            gt_entity.target_uri == pred_entity.best_candidate_uri
    )

def check_ner_matching(gt_entity, pred_entity):
    """Checks if a predicted entity matches a ground truth entity based on NER criteria."""
    return (
            gt_entity.start_position == pred_entity.start_position and
            gt_entity.end_position == pred_entity.end_position
    )