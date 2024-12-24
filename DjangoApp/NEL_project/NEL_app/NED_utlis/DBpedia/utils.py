import requests

DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"


def search_dbpedia(entity_text, dbpedia_type=None, max_results=3):
    results = fetch_dbpedia_results(entity_text, dbpedia_type)
    best_result = get_best_dbpedia_result(results)
    return best_result


def fetch_dbpedia_results(entity_text, dbpedia_type=None, max_results=3):
    """
    Fetch search results from the DBpedia Lookup API.

    :param entity_text: The text to search for.
    :param dbpedia_type: (Optional) The type of the entity.
    :param max_results: Maximum number of results to return.
    :return: The JSON response from the DBpedia Lookup API, or None if an error occurs.
    """
    params = {
        "query": entity_text,
        "format": "JSON",
        "maxResults": max_results,
    }
    if dbpedia_type:
        params["typeName"] = dbpedia_type
        params["typeNameRequired"] = "true"

    try:
        response = requests.get(DBPEDIA_LOOKUP_ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying DBpedia: {e}")
        return None


def get_best_dbpedia_result(data):
    """
    Extract the best result from the DBpedia Lookup API response.

    :param data: The JSON response from the DBpedia Lookup API.
    :return: A dictionary with the best result's details or None if no valid results are found.
    """
    if data and data.get("docs"):
        best_doc = max(data["docs"], key=lambda doc: float(doc.get("score", [0])[0]))
        return {
            "Label": best_doc.get("label", ["Unknown"])[0],
            "URI": best_doc.get("resource", [""])[0],
            "Description": best_doc.get("comment", ["No description available"])[0],
            "Score": float(best_doc.get("score", [0])[0]),
        }
    return None
