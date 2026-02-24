"""
LLM-as-a-judge evaluation for agent output quality.

Evaluates four metrics on each output:
- Relevance  (0 or 1): Was the context relevant to the post topic?
- Formatting (0 or 1): Did the output consist only of bullet points?
- Length     (0 or 1): Was the output 3-5 bullet points?
- Citation   (0 or 1): Does the output contain at least one inline citation?

Uses GPT-4o as primary and GPT-4o-mini as fallback (same OpenAI API key).
Self-contained in the eval/ directory for independent execution.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


CONTEXT_WINDOW_ERROR_SUBSTRINGS = (
    "too long",
    "context window",
    "context_length_exceeded",
    "input length",
    "maximum context",
    "prompt is too long",
    "reduce your prompt",
)


@dataclass
class JudgeResult:
    """Scores for a single evaluation case (all values are 0 or 1)."""

    relevance: float
    formatting: float
    length: float
    citation: float

    @property
    def average(self) -> float:
        return (self.relevance + self.formatting + self.length + self.citation) / 4.0


class LLMJudge:
    """
    Evaluates agent output against the three quality metrics using an LLM.

    The judge prompt instructs the LLM to return a JSON object with integer
    scores (0 or 1) for each metric, along with brief reasoning.
    """

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

    def __init__(self) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        self._primary_llm = ChatOpenAI(
            model=os.environ["OPENAI_MODEL"],
            api_key=api_key,
            max_retries=0,
        )
        self._fallback_llm = ChatOpenAI(
            model=os.environ["OPENAI_FALLBACK_MODEL"],
            api_key=api_key,
            max_retries=1,
        )
        self._config = self._load_config()

    def _load_config(self) -> dict:
        path = self.PROMPTS_DIR / "judge.yaml"
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _is_context_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in CONTEXT_WINDOW_ERROR_SUBSTRINGS)

    def _call_llm(self, user_content: str) -> str:
        """Call the judge LLM with primary/fallback logic."""
        params = self._config.get("model_params", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._config["system"]),
            ("human", self._config["user"]),
        ])

        bind_kwargs = {k: params[k] for k in ("temperature", "max_tokens", "top_p") if k in params}
        primary = self._primary_llm.bind(**bind_kwargs) if bind_kwargs else self._primary_llm
        fallback = self._fallback_llm.bind(**bind_kwargs) if bind_kwargs else self._fallback_llm

        primary_chain = prompt | primary | StrOutputParser()
        fallback_chain = prompt | fallback | StrOutputParser()

        try:
            return primary_chain.invoke({"input": user_content})
        except Exception as exc:
            if self._is_context_error(exc):
                return fallback_chain.invoke({"input": user_content})
            raise

    def evaluate(
        self,
        post_url: str,
        post_summary: str,
        actual_output: str,
    ) -> JudgeResult:
        """
        Evaluate the agent's output for a single post.

        Args:
            post_url: The Bluesky post URL (for context in the prompt).
            post_summary: A brief summary of what the post is about.
            actual_output: The bullet point explanation produced by the agent.

        Returns:
            A JudgeResult with scores for relevance, formatting, and length.
        """
        user_content = (
            f"Post URL: {post_url}\n\n"
            f"Post topic summary:\n{post_summary}\n\n"
            f"Agent output:\n{actual_output}"
        )

        raw = self._call_llm(user_content)

        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            return JudgeResult(
                relevance=float(parsed.get("relevance", 0)),
                formatting=float(parsed.get("formatting", 0)),
                length=float(parsed.get("length", 0)),
                citation=float(parsed.get("citation", 0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Warning: could not parse LLM judge response: {raw!r}")
            return JudgeResult(relevance=0.0, formatting=0.0, length=0.0, citation=0.0)
