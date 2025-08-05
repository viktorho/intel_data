import aiohttp
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import logging

logging = logging.getLogger(__name__)

class SelectionSearchTool:
    def __init__(self, search_client, *, top_k=3, browser_config=None):
        self.clients = (search_client if isinstance(search_client, list)
                        else [search_client])
        self.top_k, self.browser_config = top_k, browser_config

    async def search(self, query: str):
        for client in self.clients:
            try:
                res = await client.search(query, self.top_k)

                if res:
                    return res
            except Exception:
                continue
        return []

class BaseSearchClient(ABC):
    """All concrete clients must implement this."""

    @abstractmethod
    async def search(self, query: str, k: int) -> List[Dict]:
        """
        Return *at most* k results; each is a dict that MUST contain
        at least {"link": <url str>} but can carry extra keys.
        """

class SearxClient(BaseSearchClient):
    def __init__(self, base_url: str = "https://my-searx.example"):
        self.base = base_url.rstrip("/")

    async def search(self, query: str, k: int) -> List[Dict]:
        url = f"{self.base}/search"
        params = {"q": query, "format": "json", "num": k}

        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params, timeout=15) as r:
                data = await r.json()
        return data.get("results", [])[:k]


class BraveClient(BaseSearchClient):
    def __init__(self):
        token = os.getenv("BRAVE_API_KEY")
        if not token:
            raise RuntimeError("Set BRAVE_API_KEY env var first")
        self.headers = {"Accept": "application/json",
                        "X-Subscription-Token": token}

    async def search(self, query: str, k: int) -> List[Dict]:
        link = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": k}
        async with aiohttp.ClientSession(headers=self.headers) as s:
            logging.debug("Requesting Brave search: %s", link)
            async with s.get(link, params=params, timeout=15) as r:
                data = await r.json()
        logging.debug("Brave search response: %s", data)
        return data.get("web", {}).get("results", [])[:k]


class ExaClient(BaseSearchClient):
    def __init__(self):
        self.token = os.getenv("EXA_API_KEY")

    async def search(self, query: str, k: int) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self.token}"}
        async with aiohttp.ClientSession(headers=headers) as s:
            async with s.post("https://api.exa.ai/search",
                              json={"q": query, "num_results": k},
                              timeout=20) as r:
                data = await r.json()
        return data.get("results", [])[:k]


class TavilyClient(BaseSearchClient):
    def __init__(self):
        self.token = os.getenv("TAVILY_API_KEY")

    async def search(self, query: str, k: int) -> List[Dict]:
        url = "https://api.tavily.com/search"
        async with aiohttp.ClientSession() as s:
            async with s.post(url,
                              json={"api_key": self.token,
                                    "query": query,
                                    "num_results": k},
                              timeout=15) as r:
                data = await r.json()
        return data.get("results", [])[:k]
