"""Default LLM API settings (TokenRouter OpenAI-compatible gateway)."""

from __future__ import annotations

import os

# https://www.tokenrouter.com/docs
DEFAULT_LLM_BASE_URL = "https://api.tokenrouter.com/v1"

# TokenRouter vision models (https://www.tokenrouter.com/console/pricing)
DEFAULT_SELECT_VLM_MODEL = "google/gemini-2.5-flash-image"
DEFAULT_CAPTIONS_MODEL = "google/gemini-3.1-flash-image-preview"


def resolve_llm_api_key() -> str:
    """Return API key from env (TokenRouter / OpenAI-compatible providers)."""
    for name in ("TOKENROUTER_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


def resolve_llm_base_url(cli_base_url: str = "") -> str:
    """CLI flag < OPENAI_BASE_URL < TOKENROUTER_BASE_URL < default."""
    for candidate in (
        (os.environ.get("OPENAI_BASE_URL") or "").strip(),
        (os.environ.get("TOKENROUTER_BASE_URL") or "").strip(),
        (cli_base_url or "").strip(),
        DEFAULT_LLM_BASE_URL,
    ):
        if candidate:
            return candidate.rstrip("/")
    return DEFAULT_LLM_BASE_URL
