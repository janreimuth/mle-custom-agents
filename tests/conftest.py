"""Never let a local .env or exported credential turn regression tests into API calls."""

import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "dry-run")
    for provider in ("GROQ", "MISTRAL", "OPENROUTER", "GOOGLE", "OPENAI"):
        monkeypatch.delenv(f"{provider}_API_KEY", raising=False)
