"""Provider selection must not issue requests or silently switch execution modes."""

import os

import pytest
from pydantic_ai import models
from pydantic_ai.models.function import FunctionModel

from support import agent_limits, build_agent
from support.provider import PROVIDERS, model_from_env, model_settings


@pytest.mark.parametrize("provider", PROVIDERS)
def test_native_adapter_and_model_override(provider, monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "offline-placeholder")
    name = "openai/custom-model" if provider == "openrouter" else "custom-model"
    monkeypatch.setenv(f"{provider.upper()}_MODEL", name)
    model = model_from_env(provider)
    assert model is not None
    assert model.model_name == name
    assert not isinstance(model, FunctionModel)
    if provider == "openai":
        assert type(model).__name__ == "OpenAIResponsesModel"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_selected_provider_requires_its_own_key(provider, monkeypatch):
    monkeypatch.setenv("UNRELATED_API_KEY", "offline-placeholder")
    with pytest.raises(ValueError, match=f"{provider.upper()}_API_KEY"):
        model_from_env(provider)


def test_default_is_scripted_even_with_a_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    assert model_from_env() is None
    assert isinstance(build_agent().model, FunctionModel)


def test_bad_provider_and_blank_model_fail(monkeypatch):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        model_from_env("typo")
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    monkeypatch.setenv("GROQ_MODEL", " ")
    with pytest.raises(ValueError, match="GROQ_MODEL"):
        model_from_env("groq")


def test_configured_provider_reaches_the_agent(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    assert type(build_agent().model).__name__ == "GroqModel"


def test_openrouter_backup_is_optional_and_limits_are_bounded(monkeypatch):
    assert model_settings("openrouter")["extra_body"] == {
        "provider": {"require_parameters": True},
        "models": ["openrouter/free"],
    }
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "")
    assert model_settings("openrouter")["extra_body"] == {
        "provider": {"require_parameters": True}
    }
    assert model_settings("google")["max_retries"] == 0
    assert agent_limits().request_limit == int(os.getenv("MAX_AGENT_STEPS", "6"))
    monkeypatch.setenv("MAX_TOOL_CALLS", "0")
    with pytest.raises(ValueError, match="positive"):
        agent_limits()
