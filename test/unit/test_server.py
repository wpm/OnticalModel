import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from ontical_model.langgraph_agent import LangGraphAgent
from ontical_model.server import OnticalModelServerArgs, OnticalModelServer, app_builder
from test.schemas import NameAgeOccupation


@pytest.fixture
def custom_schema_dir():
    """Create a temporary directory with a custom schema for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir)
        schema_file = schema_path / "custom_schema.py"
        schema_file.write_text(
            '''"""Test schema for schema_path functionality."""

from pydantic import BaseModel


class CustomNameAgeOccupation(BaseModel):
    """A custom schema for testing schema_path."""

    primary: str
    secondary: str
'''
        )
        yield tmpdir


def test_ontical_model_server_args_validation():
    """Test that OnticalModelServerArgs validates required fields."""
    # Valid args
    args = OnticalModelServerArgs(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        schema_class="test.schemas.NameAgeOccupation",
    )
    assert args.model_name == "llama3.2:1b"
    assert args.base_url == "http://localhost:11434/v1"
    assert args.schema_class == "test.schemas.NameAgeOccupation"
    assert args.temperature == 0.7
    assert args.max_tokens == 150


def test_ontical_model_server_args_custom_values():
    """Test that OnticalModelServerArgs accepts custom values."""
    args = OnticalModelServerArgs(
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="sk-test123",
        schema_class="my.module.Schema",
        temperature=0.5,
        max_tokens=500,
    )
    assert args.model_name == "gpt-4"
    assert args.api_key == SecretStr("sk-test123")
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
    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
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

    # Verify LangGraphAgent was initialized with None as initial_prompt and checkpointer
    mock_agent_class.assert_called_once_with(
        mock_llm, NameAgeOccupation, initial_prompt=None, checkpointer=None
    )
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
    OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
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
    mock_agent.return_value = (
        "Response text",
        NameAgeOccupation(name="Bob", age=28, occupation="bartender"),
    )
    mock_agent_class.return_value = mock_agent

    # Create server instance
    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Mock request with valid JSON
    mock_request = Mock()
    mock_request.json = AsyncMock(return_value=["thread123", "What is your name?"])

    # Call the server
    text_response, structured_response = await server(mock_request)

    # Verify the model was called correctly
    mock_agent.assert_called_once_with("thread123", "What is your name?")
    assert text_response == "Response text"
    assert structured_response.name == "Bob"
    assert structured_response.age == 28
    assert structured_response.occupation == "bartender"


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
    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
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
        schema_class="custom_schema.CustomNameAgeOccupation",
        schema_path="/some/path",
    )
    assert args.schema_path == "/some/path"


def test_app_builder_with_schema_path(custom_schema_dir: str):
    """Test app_builder successfully imports schema from custom path."""
    # Store original sys.path to restore later
    original_path = sys.path.copy()

    try:
        args = OnticalModelServerArgs(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            schema_class="custom_schema.CustomNameAgeOccupation",
            schema_path=custom_schema_dir,
        )
        app = app_builder(args)

        # Verify it returns a Ray Serve application (deployment binding)
        assert app is not None

        # Verify schema_path was added to sys.path
        assert custom_schema_dir in sys.path
    finally:
        # Restore original sys.path
        sys.path = original_path
        # Clean up module cache
        if "custom_schema" in sys.modules:
            del sys.modules["custom_schema"]


def test_app_builder_schema_path_enables_import(custom_schema_dir: str):
    """Test that schema_path allows importing modules not in default path."""
    # Store original sys.path to restore later
    original_path = sys.path.copy()

    try:
        # First verify the module is not importable without schema_path
        if "custom_schema" in sys.modules:
            del sys.modules["custom_schema"]

        # Remove custom_schema_dir from path if it exists
        sys.path = [p for p in sys.path if p != custom_schema_dir]

        # Now use app_builder with schema_path
        args = OnticalModelServerArgs(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            schema_class="custom_schema.CustomNameAgeOccupation",
            schema_path=custom_schema_dir,
        )
        app = app_builder(args)

        # Should succeed because schema_path was added
        assert app is not None
    finally:
        # Restore original sys.path and clean up module cache
        sys.path = original_path
        if "custom_schema" in sys.modules:
            del sys.modules["custom_schema"]


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_set_thread_prompt(mock_agent_class: Mock, mock_chat_openai: Mock):
    """Test setting a thread-specific prompt."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Initially should have no thread prompts
    assert len(server.thread_prompts) == 0

    # Set a thread-specific prompt
    server.set_thread_prompt("thread123", "You are a color expert.")
    assert server.thread_prompts["thread123"] == "You are a color expert."

    # Update the same thread
    server.set_thread_prompt("thread123", "You are a helpful assistant.")
    assert server.thread_prompts["thread123"] == "You are a helpful assistant."


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_get_thread_prompt(mock_agent_class: Mock, mock_chat_openai: Mock):
    """Test getting a thread-specific prompt."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Getting a non-existent thread should return None
    assert server.get_thread_prompt("thread123") is None

    # Set and get a thread prompt
    server.set_thread_prompt("thread123", "You are a color expert.")
    assert server.get_thread_prompt("thread123") == "You are a color expert."


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_delete_thread_prompt(mock_agent_class: Mock, mock_chat_openai: Mock):
    """Test deleting a thread-specific prompt."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Set a thread prompt
    server.set_thread_prompt("thread123", "You are a color expert.")
    assert "thread123" in server.thread_prompts

    # Delete the thread prompt
    server.delete_thread_prompt("thread123")
    assert "thread123" not in server.thread_prompts

    # Deleting a non-existent prompt should not raise an error
    server.delete_thread_prompt("nonexistent")


@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
def test_list_thread_prompts(mock_agent_class: Mock, mock_chat_openai: Mock):
    """Test listing all thread prompts."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Initially should be empty
    assert server.list_thread_prompts() == {}

    # Add multiple thread prompts
    server.set_thread_prompt("thread1", "Prompt 1")
    server.set_thread_prompt("thread2", "Prompt 2")

    prompts = server.list_thread_prompts()
    assert prompts == {"thread1": "Prompt 1", "thread2": "Prompt 2"}

    # Verify it returns a copy (modifying the returned dict doesn't affect the server)
    prompts["thread3"] = "Prompt 3"
    assert "thread3" not in server.thread_prompts


@pytest.mark.asyncio
@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
async def test_ontical_model_server_call_with_thread_prompt(
    mock_agent_class: Mock, mock_chat_openai: Mock
):
    """Test that __call__ passes thread-specific prompt to the model."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent.return_value = (
        "Response text",
        NameAgeOccupation(name="Bob", age=28, occupation="bartender"),
    )
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Set a thread-specific prompt
    server.set_thread_prompt("thread123", "Custom thread prompt")

    # Mock request
    mock_request = Mock()
    mock_request.json = AsyncMock(return_value=["thread123", "What is your name?"])

    # Call the server
    await server(mock_request)

    # Verify the model was called and initial_prompt was set
    mock_agent.assert_called_once_with("thread123", "What is your name?")
    assert server.model.initial_prompt == "Custom thread prompt"


@pytest.mark.asyncio
@patch("ontical_model.server.ChatOpenAI")
@patch("ontical_model.server.LangGraphAgent")
async def test_ontical_model_server_call_without_thread_prompt(
    mock_agent_class: Mock, mock_chat_openai: Mock
):
    """Test that __call__ passes None when no thread-specific prompt is set."""
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_agent = Mock()
    mock_agent.return_value = (
        "Response text",
        NameAgeOccupation(name="Bob", age=28, occupation="bartender"),
    )
    mock_agent_class.return_value = mock_agent

    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Mock request without setting thread-specific prompt
    mock_request = Mock()
    mock_request.json = AsyncMock(return_value=["thread456", "What is your age?"])

    # Call the server
    await server(mock_request)

    # Verify the model was called and initial_prompt was set to None
    mock_agent.assert_called_once_with("thread456", "What is your age?")
    assert server.model.initial_prompt is None


def test_langgraph_agent_clear_thread_checkpoint():
    """Test clearing thread checkpoint data."""
    # Create a simple LangGraphAgent with MemorySaver
    model = Mock()
    model.invoke = Mock(return_value="text response")
    model.with_structured_output = Mock(
        return_value=Mock(
            invoke=Mock(
                return_value=NameAgeOccupation(
                    name="Bob", age=28, occupation="bartender"
                )
            )
        )
    )

    agent = LangGraphAgent(model, NameAgeOccupation)

    # Call the agent to create some checkpoint data
    agent("thread1", "Hello")
    agent("thread2", "Hi")

    # Verify checkpoints exist for both threads
    # Storage keys are nested: storage["thread_id"][checkpoint_ns][checkpoint_id]
    assert "thread1" in agent.graph.checkpointer.storage
    assert "thread2" in agent.graph.checkpointer.storage

    # Clear thread1's checkpoint
    agent.clear_thread_checkpoint("thread1")

    # Verify thread1 checkpoint is cleared but thread2 remains
    assert "thread1" not in agent.graph.checkpointer.storage
    assert "thread2" in agent.graph.checkpointer.storage


def test_server_with_redis_url():
    """Test OnticalModelServer initialization with Redis URL."""

    with patch("ontical_model.server.RedisSaver") as mock_redis_saver:
        # Mock the checkpointer
        mock_checkpointer = Mock()
        mock_redis_saver.from_conn_string.return_value = mock_checkpointer

        # Create server with redis_url
        server = OnticalModelServer.func_or_class(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            schema=NameAgeOccupation,
            redis_url="redis://localhost:6379",
        )

        # Verify RedisSaver was initialized
        mock_redis_saver.from_conn_string.assert_called_once_with(
            "redis://localhost:6379"
        )
        mock_checkpointer.setup.assert_called_once()

        # Verify server was created
        assert server.model is not None
        assert server.thread_prompts == {}


def test_server_with_redis_connection_failure():
    """Test OnticalModelServer falls back to MemorySaver when Redis fails."""

    with patch("ontical_model.server.RedisSaver") as mock_redis_saver:
        # Make Redis initialization fail
        mock_redis_saver.from_conn_string.side_effect = Exception("Connection refused")

        # Create server with redis_url (should fall back to MemorySaver)
        server = OnticalModelServer.func_or_class(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            schema=NameAgeOccupation,
            redis_url="redis://localhost:6379",
        )

        # Verify server was still created (fell back to MemorySaver)
        assert server.model is not None
        assert server.thread_prompts == {}


def test_server_without_redis_url():
    """Test OnticalModelServer initialization without Redis (uses MemorySaver)."""
    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Verify server was created with MemorySaver
    assert server.model is not None
    assert server.thread_prompts == {}
    # MemorySaver should be used (has 'storage' attribute)
    assert hasattr(server.model.graph.checkpointer, "storage")


def test_clear_thread_checkpoint_method():
    """Test OnticalModelServer.clear_thread_checkpoint method."""
    server = OnticalModelServer.func_or_class(
        model_name="llama3.2:1b",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        schema=NameAgeOccupation,
    )

    # Mock the model's clear_thread_checkpoint method
    server.model.clear_thread_checkpoint = Mock()

    # Call clear_thread_checkpoint
    server.clear_thread_checkpoint("thread123")

    # Verify it was delegated to the model
    server.model.clear_thread_checkpoint.assert_called_once_with("thread123")
