from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from hiremate_ai.config import Settings


def test_openai_profile_selects_api_settings_without_overwriting_local_settings() -> None:
    config = Settings(
        _env_file=None,
        llm_profile="openai",
        openai_api_key=SecretStr("example-key"),
        local_llm_model="gemma4:e4b",
        llm_base_url="http://host.docker.internal:11434/v1",
        poster_local_release_ollama=True,
    )
    assert config.llm_model == "gpt-6-luna"
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.llm_api_key.get_secret_value() == "example-key"
    assert config.local_llm_model == "gemma4:e4b"
    assert config.llm_temperature is None
    assert config.llm_reasoning_effort == "none"
    assert config.llm_structured_output == "function_calling"
    assert not config.poster_local_release_ollama


def test_local_profile_preserves_openai_key_but_does_not_use_it() -> None:
    config = Settings(
        _env_file=None,
        llm_profile="local",
        local_llm_base_url="http://host.docker.internal:11434/v1",
        openai_api_key=SecretStr("example-key"),
    )
    assert config.llm_model == "gemma4:e4b"
    assert config.llm_base_url == "http://host.docker.internal:11434/v1"
    assert config.llm_api_key.get_secret_value() == "ollama"
    assert config.openai_api_key.get_secret_value() == "example-key"


def test_custom_profile_keeps_existing_llm_settings() -> None:
    config = Settings(_env_file=None, llm_profile="custom", llm_model="existing-model")
    assert config.llm_model == "existing-model"


def test_langfuse_defaults_to_disabled_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LANGFUSE_TRACING_ENABLED", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    config = Settings(_env_file=None)

    assert not config.langfuse_tracing_enabled
    assert config.langfuse_base_url == "http://localhost:3000"


@pytest.mark.parametrize(
    ("key", "value"),
    [("LANGSMITH_TRACING", "true"), ("LANGSMITH_API_KEY", "test-only"), ("LANGSMITH_PROJECT", "old")],
)
def test_removed_tracing_settings_are_not_accepted(tmp_path: Path, key: str, value: str) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{key}={value}\n")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings(_env_file=dotenv)


def test_unknown_dotenv_settings_still_fail(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LANGFUSE_TRACING_ENABELD=true\n")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings(_env_file=dotenv)
