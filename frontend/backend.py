import requests
import logging

logger = logging.getLogger(__name__)

SEARCH_URL = "http://localhost:8000/search"
INGEST_URL = "http://localhost:8000/ingest"

def search_files(query: str, limit: int = 5):
    """
    GET /search?q=query&limit=limit
    """
    try:
        params = {"q": query, "limit": limit}
        headers = {"accept": "application/json"}
        response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        # The user's API returns results. Handling both list and dict-with-results key.
        if isinstance(data, list):
            return data
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return []

def ingest_directory(directory: str, rebuild: bool = True):
    """
    POST /ingest
    { "directory": "...", "rebuild": false }
    """
    try:
        payload = {"directory": directory, "rebuild": rebuild}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.post(INGEST_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return True, "Success"
    except Exception as e:
        logger.error(f"Ingest API error: {e}")
        return False, str(e)
