import sys
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from ontical_model.server import OnticalModelServerArgs, OnticalModelServer, app_builder
from test.schemas import Colors


def test_ontical_model_server_args_validation():
    """Test that OnticalModelServerArgs validates required fields."""
    # Valid args
    args = OnticalModelServerArgs(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        schema_class="test.schemas.Colors",
    )
    assert args.model_name == "llama3.2:1b"
    assert args.base_url == "http://localhost:11434/v1"
    assert args.schema_class == "test.schemas.Colors"
    assert args.schema_path is None  # default value
    assert args.api_key == "ollama"  # default value
    assert args.initial_prompt == "Answer questions accurately and succinctly."
    assert args.temperature == 0.7
    assert args.max_tokens == 150


def test_ontical_model_server_args_missing_fields():
    """Test that OnticalModelServerArgs raises ValidationError for missing fields."""
    with pytest.raises(ValidationError):
        OnticalModelServerArgs()


def test_ontical_model_server_args_custom_values():
    """Test that OnticalModelServerArgs accepts custom values."""
    args = OnticalModelServerArgs(
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="sk-test123",
        schema_class="my.module.Schema",
        initial_prompt="Custom prompt",
        temperature=0.5,
        max_tokens=500,
    )
    assert args.model_name == "gpt-4"
    assert args.api_key == "sk-test123"
    assert args.initial_prompt == "Custom prompt"
    assert args.temperature == 0.5
    assert args.max_tokens == 500


def test_app_builder_with_valid_schema(serve_config: dict[str, Any]):
    """Test app_builder successfully imports schema and creates deployment."""
    args = OnticalModelServerArgs(**serve_config)
    app = app_builder(args)

    # Verify it returns a Ray Serve application (deployment binding)
    assert app is not None
    # Ray Serve bind() returns a deployment node, which we can't easily inspect
    # but we can verify the function executes without error


def test_app_builder_with_invalid_schema():
    """Test app_builder raises error for invalid schema class."""
    args = OnticalModelServerArgs(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        schema_class="nonexistent.module.Schema",
    )

    with pytest.raises(ModuleNotFoundError):
        app_builder(args)


def test_app_builder_with_invalid_class_name():
    """Test app_builder raises error for invalid class name in valid module."""
    args = OnticalModelServerArgs(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        schema_class="test.schemas.NonexistentClass",
    )

    with pytest.raises(AttributeError):
        app_builder(args)


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_ontical_model_server_init(mock_agent_class: Mock, mock_chat_openai: Mock):
    """Test OnticalModelServer initialization."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    # Access the underlying class from the deployment
    ServerClass = OnticalModelServer.func_or_class
    server = ServerClass(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=Colors,
        initial_prompt="Test prompt",
        temperature=0.5,
        max_tokens=100,
    )

    # Verify ChatOpenAI was called with correct args
    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["model"] == "llama3.2:1b"
    assert call_kwargs["base_url"] == "http://localhost:11434/v1"
    assert call_kwargs["api_key"].get_secret_value() == "test-key"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 100

    # Verify LangGraphAgent was initialized
    mock_agent_class.assert_called_once_with(mock_llm, Colors, "Test prompt")
    assert server.model == mock_agent


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_ontical_model_server_init_with_defaults(
    mock_agent_class: Mock, mock_chat_openai: Mock
):
    """Test OnticalModelServer initialization with default values."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    # Access the underlying class from the deployment
    ServerClass = OnticalModelServer.func_or_class
    server = ServerClass(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=Colors,
        initial_prompt="Test prompt",
    )

    # Verify defaults were used
    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 150


@pytest.mark.asyncio
@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
async def test_ontical_model_server_call(
    mock_agent_class: Mock, mock_chat_openai: Mock
):
    """Test OnticalModelServer.__call__ method."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent.return_value = ("Response text", Colors(colors={"red", "white", "blue"}))
    mock_agent_class.return_value = mock_agent

    # Create server instance
    ServerClass = OnticalModelServer.func_or_class
    server = ServerClass(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=Colors,
        initial_prompt="Test prompt",
    )

    # Mock request with valid JSON
    mock_request = Mock()
    mock_request.json = AsyncMock(return_value=["thread123", "What colors?"])

    # Call the server
    text_response, structured_response = await server(mock_request)

    # Verify the model was called correctly
    mock_agent.assert_called_once_with("thread123", "What colors?")
    assert text_response == "Response text"
    assert structured_response.colors == {"red", "white", "blue"}


@pytest.mark.asyncio
@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
async def test_ontical_model_server_call_invalid_json(
    mock_agent_class: Mock, mock_chat_openai: Mock
):
    """Test OnticalModelServer.__call__ with invalid JSON."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    # Create server instance
    ServerClass = OnticalModelServer.func_or_class
    server = ServerClass(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=Colors,
        initial_prompt="Test prompt",
    )

    # Mock request with invalid JSON (not a list)
    mock_request = Mock()
    mock_request.json = AsyncMock(return_value={"thread": "123", "content": "What?"})

    # Should raise ValueError
    with pytest.raises(ValueError, match="Expected JSON array with"):
        await server(mock_request)

    # Mock request with wrong number of elements
    mock_request.json = AsyncMock(return_value=["thread123"])

    with pytest.raises(ValueError, match="Expected JSON array with"):
        await server(mock_request)


def test_ontical_model_server_args_with_schema_path():
    """Test that OnticalModelServerArgs accepts schema_path."""
    args = OnticalModelServerArgs(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        schema_class="custom_schema.CustomColors",
        schema_path="/tmp/test_schemas",
    )
    assert args.schema_path == "/tmp/test_schemas"


def test_app_builder_with_schema_path():
    """Test app_builder successfully imports schema from custom path."""
    # Store original sys.path to restore later
    original_path = sys.path.copy()

    try:
        args = OnticalModelServerArgs(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            schema_class="custom_schema.CustomColors",
            schema_path="/tmp/test_schemas",
        )
        app = app_builder(args)

        # Verify it returns a Ray Serve application (deployment binding)
        assert app is not None

        # Verify schema_path was added to sys.path
        assert "/tmp/test_schemas" in sys.path
    finally:
        # Restore original sys.path
        sys.path = original_path


def test_app_builder_schema_path_enables_import():
    """Test that schema_path allows importing modules not in default path."""
    # Store original sys.path to restore later
    original_path = sys.path.copy()

    try:
        # First verify the module is not importable without schema_path
        if "custom_schema" in sys.modules:
            del sys.modules["custom_schema"]

        # Remove /tmp/test_schemas from path if it exists
        sys.path = [p for p in sys.path if p != "/tmp/test_schemas"]

        # Now use app_builder with schema_path
        args = OnticalModelServerArgs(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            schema_class="custom_schema.CustomColors",
            schema_path="/tmp/test_schemas",
        )
        app = app_builder(args)

        # Should succeed because schema_path was added
        assert app is not None
    finally:
        # Restore original sys.path and clean up module cache
        sys.path = original_path
        if "custom_schema" in sys.modules:
            del sys.modules["custom_schema"]
