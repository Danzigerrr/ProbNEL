import requests

DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"


def search_dbpedia_by_entity_surface_form(entity_surface_form, max_results=3):
    """
    Fetch search results from the DBpedia Lookup API.

    :param entity_surface_form: The text to search for.
    :param max_results: Maximum number of results to return.
    :return: The JSON response from the DBpedia Lookup API, or None if an error occurs.
    """
    params = {
        "query": entity_surface_form,
        "format": "JSON",
        "maxResults": max_results,
    }

    try:
        response = requests.get(DBPEDIA_LOOKUP_ENDPOINT, params=params)
        response.raise_for_status()
        results = response.json()
        return results

    except requests.exceptions.RequestException as e:
        print(f"Error querying DBpedia: {e}")
        return None

