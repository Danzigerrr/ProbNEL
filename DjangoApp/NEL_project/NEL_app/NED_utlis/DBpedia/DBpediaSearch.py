import requests
from DjangoApp.NEL_project.NEL_app.NED_utlis.Candidate.Candidate import Candidate
import threading
import diskcache

class DBpediaSearch:
    """
    A class for searching DBpedia using the Lookup API with disk-based caching.
    This implementation is thread-safe and addresses memory issues.
    """
    DBPEDIA_LOOKUP_ENDPOINT = "https://lookup.dbpedia.org/api/search"
    _cache = diskcache.Cache("dbpedia_cache")  # Use diskcache for persistent and thread-safe caching
    _lock = threading.Lock() #Add a class wide lock.

    def __init__(self):
        print("DBpediaSearch disk-based caching enabled.")

    def cached_request(self, entity_surface_form, max_results=3):
        key = f"{entity_surface_form[:25]}_{max_results}"

        DBpediaSearch._lock.acquire() #acquire the lock.
        try:
            cached_result = DBpediaSearch._cache.get(key)
            if cached_result is not None:
                # print(f"Cache hit for '{entity_surface_form[:25]}'")
                return cached_result

            params = {
                "query": entity_surface_form[:25],
                "format": "JSON_FULL",
                "maxResults": max_results,
            }
            try:
                response = requests.get(DBpediaSearch.DBPEDIA_LOOKUP_ENDPOINT, params=params)
                response.raise_for_status()
                json_response = response.json()
                DBpediaSearch._cache.set(key, json_response)
                return json_response
            except requests.exceptions.RequestException as e:
                print(f"Error querying DBpedia: {e}")
                return None
        finally:
            DBpediaSearch._lock.release() #release the lock.

    def search_by_entity_surface_form(self, entity_surface_form, max_results=3):
        """
        Fetches search results using the cached request method.
        Returns a list of Candidate objects.
        """
        cached_response = self.cached_request(entity_surface_form, max_results)
        if cached_response:
            return format_candidates_list(cached_response) #Make sure you have this function defined
        return None

def print_cache_hit_or_miss_info(entity_surface_form, response):
    from_cache = "hit" if getattr(response, "from_cache", False) else "miss"
    print(f"Request for '{entity_surface_form[:25]}' {from_cache} (cached: {response.from_cache})")


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

