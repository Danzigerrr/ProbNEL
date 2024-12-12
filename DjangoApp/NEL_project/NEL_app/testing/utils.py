from django.db import transaction

from ..classes import *


def print_parsing_info(dataset):
    total_texts = len(dataset.texts)
    total_mentions = sum(len(text_obj.entities) for text_obj in dataset.texts)

    print("-" * 30)
    print("Dataset Parsing Summary:")
    print(f"Total number of texts: {total_texts}")
    print(f"Total number of entity mentions: {total_mentions}")
    print("-" * 30)


def parse_dataset_content(dataset_content, dataset_name):
    # Initialize the Dataset object
    dataset = TestDataset(dataset_name)
    # Parse each text and its entity mentions
    for text_entry in dataset_content:
        content = text_entry["text"]
        text_object = Text(content=content)  # Create a Text object

        for mention in text_entry["entity_mentions"]:
            entity_mention = FoundEntity(
                text=text_object,
                entity_label=mention["surface_form"],
                entity_type="entity",  # Placeholder, adjust as needed
                start_position=mention["position"]["start"],
                end_position=mention["position"]["end"],
                uri=mention.get("dbpedia_target_uri"),
                probabilities=[]
            )
            text_object.add_entity(entity_mention)

        dataset.add_text(text_object)
    return dataset

