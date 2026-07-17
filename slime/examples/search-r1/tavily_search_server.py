import os

from tavily import AsyncTavilyClient


async def tavily_search(query: str, top_k: int = 5, search_depth: str = "basic", api_key: str = None) -> list[dict]:
    """
    Perform a search using the Tavily API and return results in the same format
    as google_search() and local_search(): [{"document": {"contents": '"<title>"\n<text>'}}]
    """
    key = api_key or os.environ.get("TAVILY_API_KEY", "")
    client = AsyncTavilyClient(api_key=key)

    response = await client.search(
        query=query,
        max_results=top_k,
        search_depth=search_depth,
    )

    contexts = []
    for item in response.get("results", []):
        title = item.get("title", "") or "No title."
        content = item.get("content", "") or "No snippet available."
        contexts.append(
            {
                "document": {"contents": f'"{title}"\n{content}'},
            }
        )

    return contexts
