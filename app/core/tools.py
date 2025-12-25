from ddgs import DDGS


class BrowserTool:
    """Live Web Access"""

    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str) -> str:
        """Perform a safe search"""
        try:
            # Simple sync search
            results = list(self.ddgs.text(query, max_results=1))
            if results:
                return f"{results[0]['title']}: {results[0]['body']}"
            return "No relevant live data found."
        except Exception as e:
            print(f"Browser Error: {e}")
            return "Browser tool temporarily unavailable."
