from ..classes import TestDataset, Text, Entity
import json


class DatasetLoader:
    def load_dataset(self, dataset_file):
        """
        Load the dataset from a file object.

        :param dataset_file: The file object of the dataset.
        :return: A parsed TestDataset object.
        """
        dataset_content = json.load(dataset_file)
        return self.load_dataset_content(dataset_content, dataset_file.name)

    def load_dataset_content(self, dataset_content, dataset_name):
        """
        Parse the dataset content and create a TestDataset object.

        :param dataset_content: JSON content of the dataset.
        :param dataset_name: Name of the dataset file.
        :return: A TestDataset object.
        """
        dataset = TestDataset(dataset_name)

        # Parse each text and its entity mentions
        for text_entry in dataset_content:
            content = text_entry["text"]
            text_object = Text(content=content)  # Create a Text object

            for mention in text_entry["entity_mentions"]:
                entity_mention = Entity(
                    entity_label=mention["entity_label"],
                    entity_type=mention["entity_type"],
                    start_position=mention["start_position"],
                    end_position=mention["end_position"],
                    best_candidate_uri=mention.get("best_candidate_uri"),
                    probabilities=[]
                )
                text_object.add_entity(entity_mention)

            dataset.add_text(text_object)
        return dataset

    def print_dataset_info(self, dataset):
        """
        Print summary information about the dataset.

        :param dataset: A TestDataset object.
        """
        total_texts = len(dataset.texts)
        total_mentions = sum(len(text_obj.entities) for text_obj in dataset.texts)

        print("-" * 30)
        print("Dataset Parsing Summary:")
        print(f"Total number of texts: {total_texts}")
        print(f"Total number of entity mentions: {total_mentions}")
        print("-" * 30)
