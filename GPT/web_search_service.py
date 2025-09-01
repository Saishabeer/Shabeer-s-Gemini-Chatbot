from typing import List, Dict
from duckduckgo_search import DDGS


class WebSearchManager:
    """A manager to handle web searches using the free DuckDuckGo Search library."""

    def __init__(self):
        try:
            self.client = DDGS()
            self._is_enabled = True
            print("INFO: DuckDuckGo web search tool initialized successfully.")
        except Exception as e:
            self.client = None
            self._is_enabled = False
            print(f"ERROR: Failed to initialize DuckDuckGo search tool: {e}")

    def is_enabled(self) -> bool:
        """Check if the web search tool is configured and enabled."""
        return self._is_enabled

    def search(self, query: str) -> List[Dict]:
        """Performs a web search and returns a list of results in a standardized format."""
        if not self.is_enabled():
            return []
        try:
            # Fetch the top 3 results from DuckDuckGo
            results = self.client.text(query, max_results=3)
            if not results:
                return []

            # Standardize the output format for consistency across the app
            # DDGS returns: {'title': '...', 'href': '...', 'body': '...'}
            # We'll use:   {'title': '...', 'url': '...', 'content': '...'}
            return [{"title": r.get("title"), "url": r.get("href"), "content": r.get("body")} for r in results]
        except Exception as e:
            print(f"ERROR: Web search failed for query '{query}'. Reason: {e}")
            return []


# Create a single instance for the application to use
web_search_manager = WebSearchManager()