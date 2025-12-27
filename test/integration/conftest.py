"""
Fixtures for integration tests.

Integration tests assume that the Docker containers are already running.
For local development, start containers with:
    cd test/fixtures/ontical-test-llm && docker-compose up -d

For CI, the GitHub Actions workflow handles container startup/teardown.
"""

import os

import pytest
import ray
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr
from ray import serve


@pytest.fixture(scope="session")
def llm_base_url() -> str:
    """Base URL for the Ollama LLM service."""
    llm_host = os.environ.get("LLM_HOST", "localhost")
    # noinspection HttpUrlsUsage
    return f"http://{llm_host}:11434/v1"


@pytest.fixture(scope="session")
def bob_thread_ids() -> list[str]:
    """Thread IDs used for Bob's conversations in integration tests."""
    return ["bob_thread_1", "bob_thread_2", "bob_thread_3"]


@pytest.fixture(scope="session")
def bob_prompt() -> str:
    """
    Combined initial prompt for Bob in name/age/occupation tests.

    Combines the shared system prompt with Bob's specific instructions.
    """
    shared_prompt = (
        "You are participating in a brief chat conversation. Keep your responses "
        "concise and natural. When the conversation reaches a natural conclusion, "
        "exit gracefully."
    )
    bob_specific = (
        "You are Bob, a friendly person being interviewed in a chat conversation. "
        "Answer questions honestly and briefly:\n"
        "- Your name is Bob.\n"
        "- You are 28 years old.\n"
        "- You are a bartender.\n\n"
        "When the conversation naturally concludes, say goodbye and exit. "
        "Keep your responses concise and friendly."
    )
    return f"{shared_prompt}\n\n{bob_specific}"


@pytest.fixture(scope="session")
def name_age_occupation_model(llm_base_url: str) -> ChatOpenAI:
    """
    LangChain ChatOpenAI model configured to use Ollama.

    Ollama provides an OpenAI-compatible API, so we can use ChatOpenAI
    with a custom base_url pointing to the Ollama server.

    Requires Ollama to be running (via docker-compose or locally).
    """
    return ChatOpenAI(
        model="llama3.2:1b",
        base_url=llm_base_url,
        api_key=SecretStr("ollama"),  # Ollama doesn't require a real API key
        temperature=0,
        max_tokens=150,
    )


@pytest.fixture(scope="session")
def ontical_model_server_url(bob_prompt: str, bob_thread_ids: list[str]):
    """
    URL for the OnticalModelServer deployed via docker-compose.

    Sets up thread-specific initial prompts using bob_prompt by directly calling
    the set_thread_prompt method on the deployed server via Ray Serve handle.
    Cleans up by calling delete_thread_prompt on teardown.

    Requires Ray Serve to be running (via docker-compose).
    """
    url = "http://localhost:8000/"

    # Connect to the Ray cluster running in docker-compose
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True)

    try:
        # Get deployment handle for the OnticalModelServer
        # The deployment and app names come from serve_config.yaml
        handle = serve.get_deployment_handle(
            "OnticalModelServer", "ontical-test-server"
        )

        # Set up thread prompts for the thread IDs used in integration tests
        # Call set_thread_prompt method directly on the deployment
        for thread_id in bob_thread_ids:
            # Call the method and wait for it to complete with 60s timeout
            # First request may take longer as the LLM model loads
            response = handle.set_thread_prompt.remote(thread_id, bob_prompt)
            response.result(timeout_s=60)

        yield url

        # Cleanup: delete thread prompts and clear checkpoints to prevent memory leaks
        for thread_id in bob_thread_ids:
            try:
                # Delete the thread prompt
                response = handle.delete_thread_prompt.remote(thread_id)
                response.result(timeout_s=60)
                # Clear the checkpoint data to free memory
                response = handle.clear_thread_checkpoint.remote(thread_id)
                response.result(timeout_s=60)
            except Exception as e:
                # Log cleanup errors but don't fail the test
                logger.warning(f"Failed to cleanup thread {thread_id}: {e}")

    finally:
        # Disconnect from Ray cluster
        ray.shutdown()
