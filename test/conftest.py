"""
Shared fixtures for all tests.

This conftest contains only fixtures needed for unit tests that don't
require external dependencies like Docker containers.

For integration test fixtures, see test/integration/conftest.py
"""

import os
from pathlib import Path
from typing import Any

import pytest
from loguru import logger


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure loguru to use DEBUG level for all tests."""
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),  # Print to stdout
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


@pytest.fixture
def fixtures() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def llm_base_url() -> str:
    """Base URL for the LLM service (used for config generation in unit tests)."""
    llm_host = os.environ.get("LLM_HOST", "localhost")
    # noinspection HttpUrlsUsage
    return f"http://{llm_host}:11434/v1"


@pytest.fixture
def serve_config(llm_base_url: str) -> dict[str, Any]:
    """
    Generate serve config for testing OnticalModelServerArgs.

    This is used by unit tests to validate configuration parsing,
    not for actual LLM communication.
    """
    return {
        "model_name": "llama3.2:1b",
        "base_url": llm_base_url,
        "api_key": "ollama",
        "schema_class": "test.schemas.NameAgeOccupation",
        "temperature": 0.7,
        "max_tokens": 150,
    }
