import requests
from ..Candidate.Candidate import Candidate

WIKIDATA_SEARCH_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_GET_ENTITY_ENDPOINT = "https://www.wikidata.org/w/api.php"


class WikidataSearch:
    """
    A class for searching Wikidata using the API.
    """

    def __init__(self):
        """
        Initializes the WikidataSearch object.
        """
        pass

    def search_by_entity_surface_form(self, entity_surface_form, max_results=3):
        """
        Fetches search results from the Wikidata API.

        :param entity_surface_form: The text to search for.
        :param max_results: Maximum number of results to return.
        :return: A list of Candidate objects or None if an error occurs.
        """
        results = search_wikidata(entity_surface_form, max_results)
        candidates = []

        if results:
            for result in results:
                label = result.get("Label", "")
                description = result.get("Description", "")
                uri = result.get("URI", "")
                entity_id = result.get("ID", "")
                ontology_types = get_ontology_types(entity_id)
                sitelinks_popularity = count_sitelinks_popularity(entity_id)

                candidate = Candidate(
                    label=label,
                    ontology_types=ontology_types,
                    comment=description,
                    uri=uri,
                    ref_count=sitelinks_popularity
                )
                candidates.append(candidate)
            return candidates
        else:
            return []


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
                uri = f"https://www.wikidata.org/wiki/{item['id']}"
                entity_id = item['id']

                results.append({
                    "Label": label,
                    "Description": description,
                    "URI": uri,
                    "ID": entity_id,
                })

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata: {e}")
        return []



def get_entity_type(entity_id):
    """
    Fetches the type of entity (e.g., Person, Organisation) based on its 'instance of' property.
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


def get_ontology_types(wikidata_id):
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
        response = requests.get(WIKIDATA_GET_ENTITY_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        parent_types = []
        if 'entities' in data and wikidata_id in data['entities']:
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



def count_sitelinks_popularity(wikidata_id):
    """
    Calculates a popularity score for a Wikidata entity based on the number of sitelinks.
    """
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "sitelinks",
        "format": "json"
    }

    try:
        response = requests.get(WIKIDATA_GET_ENTITY_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

        if "entities" in data and wikidata_id in data["entities"]:
            entity = data["entities"][wikidata_id]
            sitelinks = entity.get("sitelinks", {})
            popularity_score = len(sitelinks)
            return popularity_score
        else:
            return 0

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitelinks for {wikidata_id}: {e}")
        return 0

