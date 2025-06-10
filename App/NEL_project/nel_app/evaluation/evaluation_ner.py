from typing import List
import json

from sympy import false

from App.NEL_project.nel_app.models.entity import Entity


class EvaluationNER:
    def __init__(self):
        self.ner_count_of_predicted_entities = 0
        self.ner_count_of_ground_truth_entities = 0
        self.ner_count_of_correctly_predicted_entities = 0
        self.ner_recall = 0.0
        self.ner_precision = 0.0
        self.ner_f1score = 0.0

    def evaluate_ner(self, predicted_entities: List[Entity], ground_truth_entities: List[Entity]):
        """Iterate through ground truth entities and evaluate if they are correctly predicted."""
        # print("$$ Iterating through Ground Truth entities:")

        for gt_entity in ground_truth_entities:
            # Increment the total number of entities
            self.ner_count_of_ground_truth_entities += 1

            for pred_entity in predicted_entities:
                # Check if predicted entity matches a ground truth entity by position
                self.ner_count_of_predicted_entities += 1
                if self.check_ner_matching(gt_entity, pred_entity):
                    self.ner_count_of_correctly_predicted_entities += 1
                    break  # Stop searching once a correct match is found

    def finalize_scores(self):
        self.calculate_ner_recall()
        self.calculate_ner_precision()
        self.calculate_ner_f1score()

    def check_ner_matching(self, gt_entity: Entity, pred_entity: Entity):
        """Checks if a predicted entity matches a ground truth entity based on NER criteria."""
        return (
                gt_entity.start_position == pred_entity.start_position and
                gt_entity.end_position == pred_entity.end_position
        )

    def calculate_ner_recall(self):
        """Calculate the recall of the NER system based on identified and total entities."""
        true_positives = self.ner_count_of_correctly_predicted_entities
        false_positives = self.ner_count_of_ground_truth_entities - true_positives

        self.ner_recall = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
    def calculate_ner_precision(self):
        """Calculate the recall of the NER system based on identified and total entities."""
        true_positives = self.ner_count_of_correctly_predicted_entities
        false_positives = self.ner_count_of_predicted_entities - true_positives

        self.ner_precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

    def calculate_ner_f1score(self):
        self.ner_f1score = (
            2 * (self.ner_precision * self.ner_recall) / (self.ner_precision + self.ner_recall)
            if (self.ner_precision + self.ner_recall)   > 0
            else 0
        )

        print(f"self.ner_f1score : {self.ner_f1score}")


    def print_results(self):
        """Print all relevant fields of the evaluation results."""
        print("\nNER evaluation Results:")
        print(f"Total number of ground truth entities: {self.ner_count_of_ground_truth_entities}")
        print(f"Total number of predicted entities: {self.ner_count_of_predicted_entities}")
        print(f"Total number of correctly identified entities: {self.ner_count_of_correctly_predicted_entities}")
        print(f"NER Recall: {self.ner_recall * 100:.2f}%")
        print(f"NER Precision: {self.ner_precision * 100:.2f}%")
        print(f"NER F1 Score: {self.ner_f1score * 100:.2f}%")

    def to_json_dict(self):
        """Convert the NER evaluation results into a JSON-serializable dictionary."""
        return {
            "total_ground_truth_entities": self.ner_count_of_ground_truth_entities,
            "correctly_identified_entities": self.ner_count_of_predicted_entities,
            "recall": round(self.ner_recall, 4)
        }

    def to_json(self):
        """Return a pretty JSON string of the NER evaluation results."""
        return json.dumps(self.to_json_dict(), indent=4)
