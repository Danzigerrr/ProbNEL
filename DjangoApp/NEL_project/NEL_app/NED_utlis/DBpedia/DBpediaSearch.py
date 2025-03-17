import requests
import re
from .DBpediaCandidate import *


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
            "format": "JSON",
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
            label = doc.get("label", [""])[0]
            ontology_types = doc.get("typeName", [])
            comment = doc.get("comment", [""])[0]
            uri = doc.get("resource", [""])[0]

            # Remove HTML tags using regular expressions
            label = re.sub(r'<[^>]+>', '', label)  # Remove HTML tags
            comment = re.sub(r'<[^>]+>', '', comment)  # Remove HTML tags

            candidate = Candidate(
                label=label,
                ontology_types=ontology_types,
                comment=comment,
                uri=uri,
                score_types_embedding_similarity=0.0,
                candidate_score=0.0
            )
            candidates.append(candidate)
    return candidates

