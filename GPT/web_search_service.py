import logging
from typing import List, Dict
from ddgs.ddgs import DDGS

logger = logging.getLogger(__name__)


class WebSearchManager:
    """A manager to handle web searches using the free DuckDuckGo Search library."""

    def __init__(self):
        # The __init__ can be very simple. We just log that it's initialized.
        # The check for whether the library works happens implicitly in the search method.
        self._is_enabled = True
        logger.info("DuckDuckGo web search manager initialized.")

    def is_enabled(self) -> bool:
        """Check if the web search tool is configured and enabled."""
        return self._is_enabled

    def search(self, query: str) -> List[Dict]:
        """Performs a web search and returns a list of results in a standardized format."""
        if not self.is_enabled():
            return []
        try:
            # Using a context manager ('with' statement) is the recommended way to use DDGS.
            # It ensures that network sessions are properly handled and closed for each search.
            with DDGS() as client:
                # Fetch the top 2 results from DuckDuckGo
                results = client.text(query, max_results=2)
                if not results:
                    return []

                # Standardize the output format for consistency across the app
                # DDGS returns: {'title': '...', 'href': '...', 'body': '...'}
                # We'll use:   {'title': '...', 'url': '...', 'content': '...'}
                return [{"title": r.get("title"), "url": r.get("href"), "content": r.get("body")} for r in results]
        except Exception as e:
            # If any error occurs (e.g., network issue, library change), we disable it for future calls
            # to avoid repeated errors, and log it.
            logger.error(f"Web search failed for query '{query}'. Disabling web search. Reason: {e}", exc_info=True)
            self._is_enabled = False
            return []


# Create a single instance for the application to use
web_search_manager = WebSearchManager()