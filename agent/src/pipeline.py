"""
Main three-step pipeline for explaining a Bluesky post.

Step 1 — Extract:   Fetch the post and summarize all content (text, images, URLs, quotes).
Step 2 — Guardrail: Check the extracted content for profanity / harmful material.
Step 3 — Explain:   Generate 3-5 bullet points of contextual background.
"""

import json
import os

from .bluesky_client import BlueskyClient
from .data_extractor import DataExtractor, ExtractedContent
from .llm_client import LLMClient
from .web_scraper import WebScraper

GUARDRAIL_BLOCKED_RESPONSE = "Post did not pass profanity filter"


class ExplainerPipeline:
    """
    Orchestrates the full Bluesky post explanation pipeline.

    Instantiates all sub-components from environment variables and wires
    them together. Call `run(post_url)` to execute the pipeline end-to-end.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._llm = LLMClient()
        self._bluesky = BlueskyClient(
            handle=os.environ["BLUESKY_HANDLE"],
            app_password=os.environ["BLUESKY_APP_PASSWORD"],
        )
        self._scraper = WebScraper()
        self._extractor = DataExtractor(
            bluesky_client=self._bluesky,
            web_scraper=self._scraper,
            llm_client=self._llm,
        )

    def run(self, post_url: str) -> str:
        """
        Execute the full pipeline on a Bluesky post URL.

        Args:
            post_url: A full bsky.app post URL.

        Returns:
            Either the 3-5 bullet point explanation, or the guardrail
            blocked message if the content failed the profanity check.
        """
        # Step 1: Extract and summarize all post content
        self._log("Step 1: Extracting post content...")
        post = self._bluesky.fetch_post(post_url)
        extracted = self._extractor.extract(post)

        if self._verbose:
            self._log(f"Extracted content:\n{extracted.combined_text}\n")

        # Step 2: Guardrail check
        self._log("Step 2: Running guardrail check...")
        if not self._passes_guardrail(extracted):
            self._log("Guardrail FAILED — returning blocked response.")
            return GUARDRAIL_BLOCKED_RESPONSE

        # Step 3: Generate contextual explanation
        self._log("Step 3: Generating explanation...")
        explanation = self._llm.call_text(
            "explanation", user_content=extracted.combined_text
        )
        return explanation

    def _passes_guardrail(self, extracted: ExtractedContent) -> bool:
        """
        Run the profanity / harmful content guardrail check.

        Calls the LLM with the guardrail prompt, which returns a JSON object
        with a "result" field set to "PASS" or "FAIL".

        Returns:
            True if the content passes, False if it fails.
        """
        raw_response = self._llm.call_text(
            "guardrail", user_content=extracted.combined_text
        )

        try:
            # Strip potential markdown fencing (```json ... ```)
            cleaned = raw_response.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            result = parsed.get("result", "").upper()
            if self._verbose and result == "FAIL":
                reason = parsed.get("reason", "no reason given")
                self._log(f"Guardrail blocked: {reason}")
            return result == "PASS"
        except (json.JSONDecodeError, AttributeError):
            # If the LLM response isn't valid JSON, treat it as a PASS
            # to avoid blocking posts due to model formatting quirks.
            # A stricter policy would return False here.
            self._log("Warning: guardrail response was not valid JSON — treating as PASS")
            return True

    def _log(self, message: str) -> None:
        """Print a message to stdout only when verbose mode is enabled."""
        if self._verbose:
            print(f"[pipeline] {message}")
