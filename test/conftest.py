import os
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@pytest.fixture
def fixtures() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def opposites_yml(fixtures: Path) -> Path:
    return fixtures / "opposites.yml"


@pytest.fixture
def llm_base_url() -> str:
    llm_host = os.environ.get("LLM_HOST", "localhost")
    return f"http://{llm_host}:11434/v1"


@pytest.fixture
def llm_model_name() -> str:
    return "llama3.2:1b"


@pytest.fixture
def chat_model(llm_base_url: str, llm_model_name: str) -> ChatOpenAI:
    """
    LangChain ChatOpenAI model configured to use Ollama.

    Ollama provides an OpenAI-compatible API, so we can use ChatOpenAI
    with a custom base_url pointing to the Ollama server.
    """
    return ChatOpenAI(
        model=llm_model_name,
        base_url=llm_base_url,
        api_key=SecretStr("ollama"),  # Ollama doesn't require a real API key
        temperature=0.7,
        max_tokens=150,  # Keep responses short for faster testing
    )
