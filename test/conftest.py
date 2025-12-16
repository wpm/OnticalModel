import os

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@pytest.fixture
def llm_base_url() -> str:
    # For local tests (non-Ray)
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


@pytest.fixture
def ontical_model_server_url() -> str:
    """
    URL for the OnticalModelServer deployed via docker-compose.

    The server is deployed using 'serve run' and is always running
    when the docker-compose stack is up.
    """
    import time
    import requests

    url = "http://localhost:8000/"

    # Wait for the server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try a simple health check
            response = requests.get(url, timeout=1)
            if response.status_code in [200, 404]:  # 404 is ok, means server is up
                return url
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                raise RuntimeError(
                    f"Failed to connect to OnticalModelServer at {url} after {max_retries} attempts"
                )
            time.sleep(1)

    return url
