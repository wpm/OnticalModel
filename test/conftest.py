import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@pytest.fixture
def fixtures() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def llm_base_url() -> str:
    llm_host = os.environ.get("LLM_HOST", "localhost")
    # noinspection HttpUrlsUsage
    return f"http://{llm_host}:11434/v1"


@pytest.fixture
def serve_config(fixtures: Path) -> dict[str, Any]:
    """Read the args section from the serve_config.yaml file."""
    config_path = fixtures / "ontical-test-llm" / "serve_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["applications"][0]["args"]


@pytest.fixture
def colors_model(llm_base_url: str, serve_config: dict[str, Any]) -> ChatOpenAI:
    """
    LangChain ChatOpenAI model configured to use Ollama.

    Ollama provides an OpenAI-compatible API, so we can use ChatOpenAI
    with a custom base_url pointing to the Ollama server.
    """

    return ChatOpenAI(
        model=serve_config["model_name"],
        base_url=llm_base_url,
        api_key=SecretStr("ollama"),  # Ollama doesn't require a real API key
        temperature=serve_config["temperature"],
        max_tokens=serve_config["max_tokens"],
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
