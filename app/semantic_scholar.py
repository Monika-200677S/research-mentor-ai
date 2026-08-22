import os
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "url",
    "externalIds",
])

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def search_papers(query: str, limit: int = 30) -> list[dict]:
    """
    Retrieve research papers from Semantic Scholar.

    Uses the bulk paper search endpoint because Semantic Scholar
    recommends it for most keyword-search use cases.

    An API key is optional. If one exists in .env, it is sent
    using the x-api-key header.
    """

    params = {
        "query": query,
        "limit": limit,
        "fields": FIELDS,
    }

    headers = {
        "User-Agent": "GapReader/1.0 Academic Research Project"
    }

    if API_KEY:
        headers["x-api-key"] = API_KEY

    max_attempts = 4

    for attempt in range(max_attempts):

        try:
            response = requests.get(
                SEARCH_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

        except requests.RequestException as e:

            if attempt == max_attempts - 1:
                raise Exception(
                    f"Could not connect to Semantic Scholar: {e}"
                )

            wait_time = 2 ** attempt
            print(
                f"Connection error. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        # -----------------------------
        # SUCCESS
        # -----------------------------
        if response.status_code == 200:

            data = response.json()

            papers = data.get("data", [])

            print(
                f"Semantic Scholar returned "
                f"{len(papers)} papers."
            )

            return papers

        # -----------------------------
        # RATE LIMIT
        # -----------------------------
        if response.status_code == 429:

            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait_time = float(retry_after)
                except ValueError:
                    wait_time = 10
            else:
                wait_time = min(
                    10 * (2 ** attempt),
                    60
                )

            # Small random delay prevents repeated
            # simultaneous retries.
            wait_time += random.uniform(0, 2)

            if attempt == max_attempts - 1:
                raise Exception(
                    "Semantic Scholar is currently "
                    "rate-limiting requests (HTTP 429). "
                    "Please wait and try again later."
                )

            print(
                f"Semantic Scholar rate limit reached. "
                f"Waiting {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)
            continue

        # -----------------------------
        # OTHER ERROR
        # -----------------------------
        raise Exception(
            f"Semantic Scholar error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    raise Exception(
        "Semantic Scholar request failed after retries."
    )