from ..classes import TestDataset, Text, OriginalEntity


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
            entity_mention = OriginalEntity(
                surface_form=mention["surface_form"],
                position_start=mention["position"]["start"],
                position_end=mention["position"]["end"],
                dbpedia_uri=mention.get("dbpedia_target_uri"),
                wikidata_uri=mention.get("wikidata_target_uri")
            )
            text_object.add_entity(entity_mention)

        dataset.add_text(text_object)
    return dataset

