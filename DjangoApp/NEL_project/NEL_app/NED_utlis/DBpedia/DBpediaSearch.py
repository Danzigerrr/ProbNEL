import requests
from DjangoApp.NEL_project.NEL_app.NED_utlis.Candidate.Candidate import *


class DBpediaSearch:
    """
    A class for searching DBpedia using the Lookup API.
    """

    DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"

    def __init__(self):
        """
        Initializes the DBPediaSearch object.
        """
        pass  # No specific initialization needed for this class

    def search_by_entity_surface_form(self, entity_surface_form, max_results=3):
        """
        Fetches search results from the DBpedia Lookup API.

        :param entity_surface_form: The text to search for.
        :param max_results: Maximum number of results to return.
        :return: The JSON response from the DBpedia Lookup API, or None if an error occurs.
        """
        params = {
            "query": entity_surface_form,
            "format": "JSON_FULL",
            "maxResults": max_results,
        }

        try:
            response = requests.get(self.DBPEDIA_LOOKUP_ENDPOINT, params=params)
            response.raise_for_status()
            list_of_candidates = format_candidates_list(response.json())
            return list_of_candidates

        except requests.exceptions.RequestException as e:
            print(f"Error querying DBpedia: {e}")
            return None


def format_candidates_list(search_results):
    """
    Extract the best result from the DBpedia Lookup API response, removing HTML tags.

    :param search_results: The JSON response from the DBpedia Lookup API.
    :return: A list of Candidate objects or None if no valid results are found.
    """
    candidates = []
    if search_results and search_results.get("docs"):
        for doc in search_results["docs"]:
            label = doc.get("label", [{}])[0].get("value", "")
            ontology_types = [item.get("value", "") for item in doc.get("typeName", [])]
            comment = doc.get("comment", [{}])[0].get("value", "")
            uri = doc.get("resource", [{}])[0].get("value", "")
            ref_count = int(doc.get("refCount", [{}])[0].get("value", "0"))

            candidate = Candidate(
                label=label,
                ontology_types=ontology_types,
                comment=comment,
                uri=uri,
                ref_count=ref_count
            )
            candidates.append(candidate)
    return candidates

