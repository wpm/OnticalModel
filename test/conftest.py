import os
from pathlib import Path

import pytest
import ray
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ray import serve

from ontical_model.schemas import Colors


@pytest.fixture
def fixtures() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def opposites_yml(fixtures: Path) -> Path:
    return fixtures / "opposites.yml"


@pytest.fixture
def llm_base_url() -> str:
    # For local tests (non-Ray)
    llm_host = os.environ.get("LLM_HOST", "localhost")
    return f"http://{llm_host}:11434/v1"


@pytest.fixture
def llm_base_url_docker() -> str:
    # For Ray server tests (Docker network)
    return "http://llm-server:11434/v1"


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


@pytest.fixture(scope="session")
def ray_serve_instance():
    """
    Connect to the external Ray Serve instance running in Docker.
    """
    import time

    # Connect to the Ray server running in Docker
    ray_address = os.environ.get("RAY_ADDRESS", "ray://localhost:10001")

    # Wait for Ray to be available
    max_retries = 30
    for i in range(max_retries):
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
        except Exception as e:
            if i == max_retries - 1:
                raise RuntimeError(
                    f"Failed to connect to Ray server at {ray_address} after {max_retries} attempts"
                ) from e
            time.sleep(1)

    yield

    # Don't shutdown the external server, just disconnect
    ray.shutdown()


@pytest.fixture
def ontical_model_server(ray_serve_instance, llm_base_url_docker, llm_model_name):
    """
    Deploy OnticalModelServer with a ChatAgent and return a handle to it.
    """
    from ontical_model.server import OnticalModelServer

    # Deploy the server with the parameters needed to create a ChatAgent
    deployment = OnticalModelServer.bind(
        model_name=llm_model_name,
        base_url=llm_base_url_docker,
        api_key="ollama",
        schema=Colors,
        initial_prompt="Answer questions accurately and succinctly.",
        temperature=0.7,
        max_tokens=150,
    )
    handle = serve.run(deployment, name="ontical-test-server", route_prefix="/")

    yield handle

    # Clean up the deployment
    try:
        serve.delete("ontical-test-server")
    except Exception:
        pass  # Ignore errors if server is already stopped
