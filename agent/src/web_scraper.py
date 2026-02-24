"""
Web scraper for extracting readable text content from URLs.
"""

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class ScrapedPage:
    """Result of scraping a URL."""

    url: str
    title: Optional[str]
    body_text: str
    fetch_error: Optional[str] = None


class WebScraper:
    """
    Fetches and extracts readable plaintext from a URL.

    Strips HTML boilerplate (scripts, styles, navigation, footers) and
    prefers semantic content tags (<article>, <main>) over the raw body.
    Truncates to MAX_TEXT_CHARS to avoid overflowing the LLM context window.
    """

    TIMEOUT_SECONDS: int = 10
    MAX_TEXT_CHARS: int = 8_000

    HEADERS: dict = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; bluesky-explainer/1.0; "
            "+https://github.com/user/bluesky-explainer)"
        )
    }

    BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]

    def scrape(self, url: str) -> ScrapedPage:
        """
        Fetch and extract text from a URL.

        On any network or HTTP error, returns a ScrapedPage with fetch_error
        set and an empty body_text. Callers should handle this gracefully.

        Args:
            url: The URL to scrape.

        Returns:
            A ScrapedPage with the extracted text (or error details).
        """
        try:
            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=self.TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
            return self._parse_html(url, response.text)
        except requests.RequestException as exc:
            return ScrapedPage(url=url, title=None, body_text="", fetch_error=str(exc))

    def _parse_html(self, url: str, html: str) -> ScrapedPage:
        """
        Parse HTML and extract cleaned plaintext.

        Removes boilerplate tags, prefers <article> or <main> for content,
        collapses whitespace, and truncates to MAX_TEXT_CHARS.
        """
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(self.BOILERPLATE_TAGS):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else None

        content_tag = soup.find("article") or soup.find("main") or soup.find("body")
        raw_text = content_tag.get_text(separator=" ", strip=True) if content_tag else soup.get_text(separator=" ", strip=True)

        cleaned = " ".join(raw_text.split())
        truncated = cleaned[: self.MAX_TEXT_CHARS]

        return ScrapedPage(url=url, title=title, body_text=truncated)
