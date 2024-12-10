import requests

DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"


def search_dbpedia(entity_text, dbpedia_type=None, max_results=3):
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
        data = response.json()

        if data.get("docs"):
            best_doc = max(data["docs"], key=lambda doc: float(doc.get("score", [0])[0]))
            return {
                "Label": best_doc.get("label", ["Unknown"])[0],
                "URI": best_doc.get("resource", [""])[0],
                "Description": best_doc.get("comment", ["No description available"])[0],
                "Score": float(best_doc.get("score", [0])[0]),
            }
    except requests.exceptions.RequestException as e:
        print(f"Error querying DBpedia: {e}")

    return None
