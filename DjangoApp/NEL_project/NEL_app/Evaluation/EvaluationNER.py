from typing import List
import json
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity


class EvaluationNER:
    def __init__(self):
        self.ner_count_of_identified_entities = 0
        self.ner_count_of_total_number_of_entities = 0
        self.ner_accuracy = 0.0

    def evaluate_ner(self, predicted_entities: List[Entity], ground_truth_entities: List[Entity]):
        """Iterate through ground truth entities and evaluate if they are correctly predicted."""
        print("$$ Iterating through Ground Truth entities:")

        for gt_entity in ground_truth_entities:
            # Increment the total number of entities
            self.ner_count_of_total_number_of_entities += 1

            for pred_entity in predicted_entities:
                # Check if predicted entity matches a ground truth entity by position
                if self.check_ner_matching(gt_entity, pred_entity):
                    self.ner_count_of_identified_entities += 1
                    break  # Stop searching once a correct match is found

        self.calculate_ner_accuracy()

        self.print_results()

    def check_ner_matching(self, gt_entity: Entity, pred_entity: Entity):
        """Checks if a predicted entity matches a ground truth entity based on NER criteria."""
        return (
                gt_entity.start_position == pred_entity.start_position and
                gt_entity.end_position == pred_entity.end_position
        )

    def calculate_ner_accuracy(self):
        """Calculate the accuracy of the NER system based on identified and total entities."""
        self.ner_accuracy = (
            self.ner_count_of_identified_entities / self.ner_count_of_total_number_of_entities
            if self.ner_count_of_total_number_of_entities > 0
            else 0
        )

    def print_results(self):
        """Print all relevant fields of the evaluation results."""
        print("\nNER Evaluation Results:")
        print(f"Total number of ground truth entities: {self.ner_count_of_total_number_of_entities}")
        print(f"Total number of correctly identified entities: {self.ner_count_of_identified_entities}")
        print(f"NER Accuracy: {self.ner_accuracy * 100:.2f}%")

    def to_json_dict(self):
        """Convert the NER evaluation results into a JSON-serializable dictionary."""
        return {
            "total_ground_truth_entities": self.ner_count_of_total_number_of_entities,
            "correctly_identified_entities": self.ner_count_of_identified_entities,
            "accuracy": round(self.ner_accuracy, 4)
        }

    def to_json(self):
        """Return a pretty JSON string of the NER evaluation results."""
        return json.dumps(self.to_json_dict(), indent=4)
