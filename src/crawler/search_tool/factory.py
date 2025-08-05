# search_factory.py
from typing import Sequence

from .adapter import (
    SearxClient, BraveClient, ExaClient, TavilyClient, SelectionSearchTool
)

NAME_TO_CLASS = {
    "searx":  SearxClient,
    "brave":  BraveClient,
    "exa":    ExaClient,
    "tavily": TavilyClient,
}

def build_search_tool(
    names: str | Sequence[str],
    *,
    top_k: int = 3,
    browser_config: dict | None = None,
) -> SelectionSearchTool:
    """
    names : either a single provider ID or a comma / list of them.
            Order matters → ['searx', 'brave'] = try SearXNG then Brave.
    """
    if isinstance(names, str):
        names = [n.strip().lower() for n in names.split(",") if n.strip()]

    clients = []
    for n in names:
        try:
            clients.append(NAME_TO_CLASS[n]())      # call its constructor
        except KeyError:
            raise ValueError(
                f"Unknown provider '{n}'. Allowed: {list(NAME_TO_CLASS)}")
    return SelectionSearchTool(
        search_client=clients,
        top_k=top_k,
        browser_config=browser_config,
    )
