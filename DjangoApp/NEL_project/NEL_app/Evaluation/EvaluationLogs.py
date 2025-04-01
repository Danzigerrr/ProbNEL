import os
import csv
import json
from datetime import datetime


class EvaluationLogs:
    def __init__(self):
        self.logs = []


    def create_logs(self, predicted_entities, ground_truth_entities):
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
                self.create_log_entry(gt_entity, matched_pred_entity, True, 0)
            else:
                print(f"$ No Correct prediction found for '{gt_entity.entity_label}'({gt_entity.start_position}, {gt_entity.end_position})")
                entity_with_cand, candidate_index = self.find_candidate_index(gt_entity, predicted_entities)
                if entity_with_cand is not None:
                    self.create_log_entry(gt_entity, entity_with_cand, False, candidate_index)


    def create_log_entry(self, gt_entity, matched_pred_entity, if_correct_prediction, matching_candidate_index):
        log_entry = {
            "entity_label": gt_entity.entity_label,
            "start_position": gt_entity.start_position,
            "end_position": gt_entity.end_position,
            "correct_prediction": if_correct_prediction,
            "matching_candidate_index": matching_candidate_index,
            "candidates": [(c.label, c.score_types_embeddings_similarity, c.score_levenshtein_distance,
                            c.score_popularity, c.score_context) for c in matched_pred_entity.candidates]
        }
        self.logs.append(log_entry)
        print(f"Logs updated ({len(self.logs)})")

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

    def save_logs_to_files(self):
        os.makedirs("Evaluation_Logs", exist_ok=True)  # Ensure directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_logs_{timestamp}"
        self.save_logs_to_json(output_filename)
        self.save_logs_to_csv(output_filename)

    def save_logs_to_csv(self, output_filename):
        """Saves the logs to a CSV file."""

        with open(f"Evaluation_Logs/{output_filename}.csv", mode='w', newline='', encoding='utf-16') as file:
            writer = csv.writer(file)
            writer.writerow(["entity_label", "start_position", "end_position", "correct_prediction", "candidates", "matching_candidate_index"])
            for log in self.logs:
                candidates_str = json.dumps(log["candidates"])
                writer.writerow([log["entity_label"], log["start_position"], log["end_position"], log["correct_prediction"], candidates_str, log.get("matching_candidate_index", None)])

    def save_logs_to_json(self, output_filename):
        """Saves the logs to a JSON file."""
        with open(f"Evaluation_Logs/{output_filename}.json", 'w', encoding='utf-16') as file:
            json.dump(self.logs, file, indent=4)

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