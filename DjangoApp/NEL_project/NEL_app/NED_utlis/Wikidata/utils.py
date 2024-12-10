import requests

WIKIDATA_SEARCH_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_GET_ENTITY_ENDPOINT = "https://www.wikidata.org/w/api.php"


def search_wikidata(entity_text, max_results=3):
    """
    Query Wikidata API to retrieve matching entities based on search text.
    """
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "format": "json",
        "language": "en",
        "uselang": "en",
        "limit": max_results,
    }

    try:
        response = requests.get(WIKIDATA_SEARCH_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        if data.get('search'):
            for item in data['search']:
                label = item['label']
                description = item.get('description', 'No description available')
                url = f"https://www.wikidata.org/wiki/{item['id']}"
                entity_id = item['id']

                # Now, we fetch detailed information about the entity to get its type (instance of)
                type_info = get_entity_type(entity_id)

                results.append({
                    "Label": label,
                    "Description": description,
                    "URL": url,
                    "ID": entity_id,
                    "Type": type_info
                })

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata: {e}")
        return []


def get_entity_type(entity_id):
    """
    Fetches the type of an entity (e.g., Person, Organisation) based on its 'instance of' property.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "sites": "wikidata",
        "props": "claims",
        "format": "json",
    }

    try:
        response = requests.get(WIKIDATA_GET_ENTITY_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for the 'instance of' (P31) claim to determine the type of the entity
        if "entities" in data and entity_id in data["entities"]:
            entity = data["entities"][entity_id]
            claims = entity.get("claims", {})
            if "P31" in claims:
                # 'P31' is the property for "instance of", which typically identifies the entity's type
                entity_type = claims["P31"][0]["mainsnak"]["datavalue"]["value"]["id"]
                # Return the type label from the corresponding Wikidata entity
                type_label = get_entity_label(entity_type)
                return type_label

        return "Unknown"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching entity type for {entity_id}: {e}")
        return "Unknown"


def get_entity_label(entity_id):
    """
    Fetch the label of a Wikidata entity based on its ID.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "labels",
        "languages": "en"
    }

    try:
        response = requests.get(WIKIDATA_GET_ENTITY_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        if "entities" in data and entity_id in data["entities"]:
            return data["entities"][entity_id]["labels"]["en"]["value"]

        return "Unknown"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching label for entity type {entity_id}: {e}")
        return "Unknown"


def get_parent_types(wikidata_id):
    """
    Query Wikidata to fetch the parent classes (hierarchy) of a given entity type.
    """
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "claims",
        "format": "json"
    }

    try:
        response = requests.get(WIKIDATA_SEARCH_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        parent_types = []
        entity = data['entities'][wikidata_id]

        # Check if the entity has the 'P31' property (instance of)
        if 'claims' in entity and 'P31' in entity['claims']:
            for claim in entity['claims']['P31']:
                parent_id = claim['mainsnak']['datavalue']['value']['id']
                # Fetch the label for each parent ID to make it human-readable
                parent_label = get_entity_label(parent_id)
                parent_types.append(parent_label)

        # Return parent types as a list of human-readable labels
        return parent_types

    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata for parent types of {wikidata_id}: {e}")
        return []


if __name__ == "__main__":
    # NER output from Flair (mocked here for demonstration purposes)
    sentence = "Notre Dame, the iconic medieval cathedral in Paris, reopens after five years of speedy reconstruction work."
    ner_spans = [
        {"text": "Notre Dame", "type": "FAC", "score": 1.0000},
        {"text": "Paris", "type": "GPE", "score": 1.0000},
        {"text": "five years", "type": "DATE", "score": 1.0000},
    ]

    for span in ner_spans:
        entity_text = span["text"]
        print(f"Disambiguating entity: {entity_text}")
        results = search_wikidata(entity_text)

        if results:
            for result in results:
                print(f"Best match for '{entity_text}':")
                print(f"Label: {result['Label']}")
                print(f"Description: {result['Description']}")
                print(f"URL: {result['URL']}")
                print(f"Type: {result['Type']}")
                print(f"Parent types (hierarchy from detailed to general):")

                # Fetch parent types
                wikidata_id = result['ID']  # Wikidata ID from the first result
                parent_types = get_parent_types(wikidata_id)
                # Traverse and print parent types (you can implement a recursive call if necessary to get the full hierarchy)
                while parent_types:
                    parent_type_id = parent_types.pop(0)  # Get the next parent type
                    print(f"- {parent_type_id}")  # Print parent type ID
                print("\n")
        else:
            print(f"No matches found for '{entity_text}'\n")
        print("----------\n")
