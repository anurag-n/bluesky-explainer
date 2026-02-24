"""
LLM client abstraction using LangChain LCEL.

Primary model:  GPT-4o (OpenAI) — used for all text and vision calls.
Fallback model: GPT-4o-mini (OpenAI) — triggered on context-window errors.

All prompts are loaded from YAML files in the agent/prompts/ directory.
"""

import base64
import os
from pathlib import Path
from typing import Any

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


# Error substrings that indicate the context window was exceeded.
# When caught in call_text or call_vision, the pipeline retries with the fallback.
CONTEXT_WINDOW_ERROR_SUBSTRINGS = (
    "too long",
    "context window",
    "context_length_exceeded",
    "input length",
    "maximum context",
    "prompt is too long",
    "reduce your prompt",
)


class PromptConfig:
    """
    Loads and parses a YAML prompt configuration file.

    Expected YAML structure:
        system: |
          System message text here.
        user: |
          {input}
        model_params:
          temperature: 0.3
          max_tokens: 1024
    """

    def __init__(self, yaml_path: Path) -> None:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.system_template: str = data["system"]
        self.user_template: str = data.get("user", "{input}")
        self.model_params: dict[str, Any] = data.get("model_params", {})


class LLMClient:
    """
    LangChain-based LLM client with a primary GPT-4o and a GPT-4o-mini fallback.

    Both models support text and vision, so the same primary/fallback pair is
    used for all call types. The fallback is triggered only on context-window
    errors; all other exceptions propagate normally.

    Usage:
        client = LLMClient()
        result = client.call_text("explanation", user_content="...")
        result = client.call_vision("image_summary", image_bytes=b"...", mime_type="image/jpeg")
    """

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

    def __init__(self) -> None:
        self._primary_model = os.environ["OPENAI_MODEL"]
        self._fallback_model = os.environ["OPENAI_FALLBACK_MODEL"]
        self._primary_llm = self._build_llm(self._primary_model)
        self._fallback_llm = self._build_llm(self._fallback_model)

    def _build_llm(self, model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
        )

    def _load_config(self, prompt_name: str) -> PromptConfig:
        """Load a prompt YAML file by its stem name (e.g. 'explanation')."""
        path = self.PROMPTS_DIR / f"{prompt_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return PromptConfig(path)

    def _is_context_error(self, exc: Exception) -> bool:
        """Check if an exception indicates a context window overflow."""
        msg = str(exc).lower()
        return any(substr in msg for substr in CONTEXT_WINDOW_ERROR_SUBSTRINGS)

    def _apply_model_params(self, llm: ChatOpenAI, params: dict[str, Any]) -> ChatOpenAI:
        """Return a copy of the LLM with model_params bound (temperature, max_tokens, top_p)."""
        bind_kwargs = {
            k: params[k]
            for k in ("temperature", "max_tokens", "top_p")
            if k in params
        }
        return llm.bind(**bind_kwargs) if bind_kwargs else llm

    def call_text(self, prompt_name: str, user_content: str) -> str:
        """
        Run a text-based LLM call using a named prompt config.

        Builds an LCEL chain: ChatPromptTemplate | LLM | StrOutputParser.
        Falls back to the smaller model if a context-window error is raised.

        Args:
            prompt_name: Stem of the YAML file in agent/prompts/ (e.g. "explanation").
            user_content: Content to fill the {input} placeholder.

        Returns:
            The LLM's text response.
        """
        config = self._load_config(prompt_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", config.system_template),
            ("human", config.user_template),
        ])

        primary = self._apply_model_params(self._primary_llm, config.model_params)
        fallback = self._apply_model_params(self._fallback_llm, config.model_params)

        primary_chain: Runnable = prompt | primary | StrOutputParser()
        fallback_chain: Runnable = prompt | fallback | StrOutputParser()

        try:
            return primary_chain.invoke({"input": user_content})
        except Exception as exc:
            if self._is_context_error(exc):
                return fallback_chain.invoke({"input": user_content})
            raise

    def call_vision(
        self,
        prompt_name: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Run a vision LLM call to describe or summarize an image.

        Constructs a multimodal HumanMessage with the base64-encoded image.
        Both the primary (GPT-4o) and fallback (GPT-4o-mini) support vision
        via the same image_url message format.

        Args:
            prompt_name: Stem of the YAML file in agent/prompts/ (e.g. "image_summary").
            image_bytes: Raw bytes of the image.
            mime_type: MIME type of the image (default: "image/jpeg").

        Returns:
            The LLM's text description of the image.
        """
        config = self._load_config(prompt_name)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        messages = [
            SystemMessage(content=config.system_template),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": config.user_template.replace("{input}", "")},
                ]
            ),
        ]

        primary = self._apply_model_params(self._primary_llm, config.model_params)
        fallback = self._apply_model_params(self._fallback_llm, config.model_params)

        try:
            response = primary.invoke(messages)
            return StrOutputParser().invoke(response)
        except Exception as exc:
            if self._is_context_error(exc):
                response = fallback.invoke(messages)
                return StrOutputParser().invoke(response)
            raise
