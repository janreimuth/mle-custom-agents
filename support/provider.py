"""Select a native PydanticAI model without changing the agent's tools."""

import os

from pydantic_ai.models import Model, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

PROVIDERS = {
    "groq": ("groq", "openai/gpt-oss-20b"),
    "mistral": ("mistral", "ministral-14b-latest"),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "google": ("google", "gemma-4-26b-a4b-it"),
    "openai": ("openai", "gpt-6-luna"),
}


def model_from_env(provider: str | None = None) -> Model | None:
    """Require an explicit live provider and its key; never silently simulate."""

    provider = (provider or os.getenv("LLM_PROVIDER", "dry-run")).strip().lower()
    if provider == "dry-run":
        return None
    if provider not in PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER must be dry-run or one of {', '.join(PROVIDERS)}."
        )
    if not os.getenv(f"{provider.upper()}_API_KEY", "").strip():
        raise ValueError(f"Set {provider.upper()}_API_KEY for LLM_PROVIDER={provider}.")
    prefix, default = PROVIDERS[provider]
    name = os.getenv(f"{provider.upper()}_MODEL", default).strip()
    if not name:
        raise ValueError(f"{provider.upper()}_MODEL must not be blank.")
    return infer_model(f"{prefix}:{name}")


def model_settings(provider: str | None = None) -> ModelSettings:
    """Bound each request and keep OpenRouter's backup within that provider."""

    settings = ModelSettings(timeout=30, max_tokens=2048, max_retries=0)
    provider = (provider or os.getenv("LLM_PROVIDER", "dry-run")).strip().lower()
    if provider == "openrouter":
        backup = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free").strip()
        settings["extra_body"] = {
            "provider": {"require_parameters": True},
            **({"models": [backup]} if backup else {}),
        }
    return settings


def agent_limits() -> UsageLimits:
    """Use separate positive budgets for model requests and application tools."""

    requests = int(os.getenv("MAX_AGENT_STEPS", "6"))
    tools = int(os.getenv("MAX_TOOL_CALLS", "5"))
    if requests < 1 or tools < 1:
        raise ValueError("MAX_AGENT_STEPS and MAX_TOOL_CALLS must be positive.")
    return UsageLimits(request_limit=requests, tool_calls_limit=tools)
