"""
Data extraction orchestrator for Bluesky posts.

Extracts text, images, referenced URLs, and quoted posts from a BlueskyPost,
then uses the LLM to summarize and translate each piece into a unified
text representation for downstream pipeline steps.
"""

from dataclasses import dataclass, field
from typing import Optional

from .bluesky_client import BlueskyClient, BlueskyPost
from .llm_client import LLMClient
from .web_scraper import WebScraper


@dataclass
class ExtractedContent:
    """
    Aggregated, summarized content extracted from a Bluesky post.

    All text is in English (translated if necessary) and combined into
    a single `combined_text` field for LLM consumption.
    """

    post_uri: str
    post_text_summary: str
    image_summaries: list[str] = field(default_factory=list)
    external_url_summary: Optional[str] = None
    quoted_post_summary: Optional[str] = None

    @property
    def combined_text(self) -> str:
        """
        Build a single string with all extracted content sections.

        The combined text is used as the user prompt for the guardrail
        check and the final explanation step.
        """
        parts: list[str] = []

        if self.post_text_summary:
            parts.append(f"Post content:\n{self.post_text_summary}")

        for i, summary in enumerate(self.image_summaries, start=1):
            parts.append(f"Image {i} description:\n{summary}")

        if self.external_url_summary:
            parts.append(f"Referenced URL content:\n{self.external_url_summary}")

        if self.quoted_post_summary:
            parts.append(f"Quoted post content:\n{self.quoted_post_summary}")

        return "\n\n".join(parts)


class DataExtractor:
    """
    Orchestrates all extraction and summarization for a Bluesky post.

    Extraction steps:
    1. Summarize (and translate if needed) the main post text.
    2. For each image: run a vision LLM call to describe the image.
    3. For an external URL: scrape the page and summarize the text.
    4. For a quoted post: fetch it via the Bluesky API and summarize.

    All summaries are combined into an ExtractedContent object.
    """

    def __init__(
        self,
        bluesky_client: BlueskyClient,
        web_scraper: WebScraper,
        llm_client: LLMClient,
    ) -> None:
        self._bluesky = bluesky_client
        self._scraper = web_scraper
        self._llm = llm_client

    def extract(self, post: BlueskyPost) -> ExtractedContent:
        """
        Extract and summarize all content from a BlueskyPost.

        Args:
            post: A populated BlueskyPost dataclass.

        Returns:
            An ExtractedContent object with all summaries combined.
        """
        content = ExtractedContent(post_uri=post.uri, post_text_summary="")

        # Step 1: Summarize main post text (translate if non-English)
        if post.text.strip():
            content.post_text_summary = self._llm.call_text(
                "summarization", user_content=post.text
            )
        else:
            content.post_text_summary = "(No text in post)"

        # Step 2: Summarize images via vision LLM
        for image in post.images:
            if image.image_bytes:
                summary = self._llm.call_vision(
                    "image_summary",
                    image_bytes=image.image_bytes,
                    mime_type=image.mime_type,
                )
            elif image.alt_text:
                # Fall back to alt text if blob fetch failed
                summary = f"[Image could not be loaded. Alt text: {image.alt_text}]"
            else:
                summary = "[Image could not be loaded and has no alt text]"
            content.image_summaries.append(summary)

        # Step 3: Scrape and summarize external URL
        if post.external_url:
            scraped = self._scraper.scrape(post.external_url)
            if scraped.fetch_error:
                fallback_parts = []
                if post.external_title:
                    fallback_parts.append(f"Title: {post.external_title}")
                if post.external_description:
                    fallback_parts.append(f"Description: {post.external_description}")
                if fallback_parts:
                    source_text = " | ".join(fallback_parts)
                else:
                    source_text = f"[URL could not be fetched: {post.external_url}]"
            else:
                page_title = f"Page title: {scraped.title}\n\n" if scraped.title else ""
                source_text = page_title + scraped.body_text

            if source_text and not source_text.startswith("["):
                content.external_url_summary = self._llm.call_text(
                    "summarization", user_content=source_text
                )
            else:
                content.external_url_summary = source_text

        # Step 4: Fetch and summarize quoted post
        if post.quoted_post_uri:
            try:
                quoted = self._bluesky.fetch_post_by_uri(post.quoted_post_uri)
                if quoted.text.strip():
                    content.quoted_post_summary = self._llm.call_text(
                        "summarization", user_content=quoted.text
                    )
                else:
                    content.quoted_post_summary = "(Quoted post has no text)"
            except Exception as exc:
                content.quoted_post_summary = f"[Quoted post could not be fetched: {exc}]"

        return content
