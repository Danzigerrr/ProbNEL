from .classes.Dataset import *
from .classes.Text import *
from .classes.EntityMention import *


def print_parsing_info(dataset):
    total_texts = len(dataset.texts)
    total_mentions = sum(len(text_obj.entity_mentions) for text_obj in dataset.texts)

    # Print general statistics
    print("-"*30)
    print("Dataset Parsing Summary:")
    print(f"Total number of texts: {total_texts}")
    print(f"Total number of entity mentions: {total_mentions}")
    print("-"*30)


def parse_dataset_content(dataset_content):
    # Initialize the Dataset object
    dataset = Dataset()
    # Parse each text and its entity mentions
    for text_entry in dataset_content:
        text_content = text_entry["text"]
        entity_mentions = []

        for mention in text_entry["entity_mentions"]:
            entity_mention = EntityMention(
                surface_form=mention["surface_form"],
                position_start=mention["position"]["start"],
                position_end=mention["position"]["end"],
                dbpedia_uri=mention["dbpedia_target_uri"],
                wikidata_uri=mention["wikidata_target_uri"]
            )
            entity_mentions.append(entity_mention)

        # Create a Text object and add it to the Dataset
        text_object = Text(text=text_content, entity_mentions=entity_mentions)
        dataset.add_text(text_object)
    return dataset
