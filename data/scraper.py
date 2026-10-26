"""
Web Scraper Utility for News Articles
Extracts title, body text, author, and metadata from live article URLs.
"""

import re
import urllib.parse
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup


class NewsScraper:
    """Scrapes news articles from web URLs."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def scrape_article(self, url: str) -> Dict[str, Optional[str]]:
        """
        Scrapes and extracts headline, text, author, and publication date from a URL.

        Args:
            url: Valid web URL of a news article.

        Returns:
            dict containing:
                - title: Extracted article title
                - text: Extracted article body text
                - full_text: Combined title and body for classification
                - url: Original URL
                - error: Error message if scraping failed, else None
        """
        result = {
            "title": "",
            "text": "",
            "full_text": "",
            "url": url,
            "error": None,
        }

        # Basic URL validation
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            result["error"] = "Invalid URL format. Please include http:// or https://"
            return result

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except requests.exceptions.RequestException as e:
            result["error"] = f"Network error fetching article: {str(e)}"
            return result

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted tags
            for unwanted in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
                unwanted.decompose()

            # 1. Extract Title
            title = ""
            # Try meta open-graph title first
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            else:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text(strip=True)

            # 2. Extract Body Text
            body_paragraphs = []
            
            # Check for common article container tags
            article_tag = soup.find("article") or soup.find("div", class_=re.compile(r"article[-_]body|entry[-_]content|story-body|post-content", re.I))
            
            if article_tag:
                paras = article_tag.find_all("p")
            else:
                paras = soup.find_all("p")

            for p in paras:
                text = p.get_text(strip=True)
                # Filter out short disclaimers, cookie notices, or social share prompts
                if len(text.split()) > 4 and not re.search(r"cookie|subscribe|all rights reserved|terms of service", text, re.I):
                    body_paragraphs.append(text)

            body_text = "\n\n".join(body_paragraphs)

            if not body_text and title:
                body_text = title

            result["title"] = title
            result["text"] = body_text
            result["full_text"] = f"{title}\n\n{body_text}".strip()

            if not result["full_text"]:
                result["error"] = "Could not extract readable article text from this page."

        except Exception as e:
            result["error"] = f"Parsing error: {str(e)}"

        return result


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Machine_learning"
    scraper = NewsScraper()
    print(f"Testing scraper on: {test_url}")
    data = scraper.scrape_article(test_url)
    if data["error"]:
        print("Error:", data["error"])
    else:
        print("Title:", data["title"])
        print(f"Content length: {len(data['text'])} chars")
        print("Snippet:", data["text"][:300] + "...")
