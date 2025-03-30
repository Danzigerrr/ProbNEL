import requests
import json
from functools import lru_cache
from DjangoApp.NEL_project.NEL_app.NED_utlis.Candidate.Candidate import Candidate


class DBpediaSearch:
    """
    A class for searching DBpedia using the Lookup API with caching.
    """

    DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"

    def __init__(self):
        """Initializes the DBPediaSearch object."""
        pass  # No specific initialization needed for this class

    @staticmethod
    @lru_cache(maxsize=500)  # Cache up to 500 unique queries
    def cached_request(entity_surface_form, max_results=3):
        """
        Cached method to fetch search results from the DBpedia Lookup API.

        :param entity_surface_form: The text to search for.
        :param max_results: Maximum number of results to return.
        :return: The formatted list of Candidate objects or None if an error occurs.
        """
        params = {
            "query": entity_surface_form[:25],  # Limit to 25 characters
            "format": "JSON_FULL",
            "maxResults": max_results,
        }

        try:
            response = requests.get(DBpediaSearch.DBPEDIA_LOOKUP_ENDPOINT, params=params)
            response.raise_for_status()
            return json.dumps(response.json())  # Convert to a hashable string
        except requests.exceptions.RequestException as e:
            print(f"Error querying DBpedia: {e}")
            return None

    def search_by_entity_surface_form(self, entity_surface_form, max_results=3):
        """
        Fetches search results using the cached request method.

        :param entity_surface_form: The text to search for.
        :param max_results: Maximum number of results to return.
        :return: The list of Candidate objects or None if an error occurs.
        """
        cached_response = self.cached_request(entity_surface_form, max_results)
        if cached_response:
            return format_candidates_list(json.loads(cached_response))  # Convert back to dict
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

