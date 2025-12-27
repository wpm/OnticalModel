import importlib
import sys
from typing import Type, Generic, TypeVar, Optional, Annotated

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import RedisSaver
from pydantic import BaseModel, SecretStr, field_validator
from ray import serve
from starlette.requests import Request

from ontical_model.langgraph_agent import LangGraphAgent

SCHEMA = TypeVar("SCHEMA", bound=BaseModel)


class OnticalModelServerArgs(BaseModel):
    """
    Configuration arguments for OnticalModelServer deployment.
    """

    model_name: Annotated[str, "The name of the LLM model to use (e.g., 'llama3.2:1b')"]
    base_url: Annotated[str, "The base URL for the OpenAI-compatible API"]
    api_key: Annotated[
        str | SecretStr, "The API key (can be a dummy value for Ollama)"
    ] = SecretStr("")
    schema_class: Annotated[
        str,
        "Fully qualified name of the Pydantic schema class (e.g., 'test.schemas.Colors')",
    ]
    schema_path: Annotated[
        Optional[str],
        "Optional path to directory containing schema module. "
        "If provided, this path will be added to sys.path before importing schema_class.",
    ] = None
    redis_url: Annotated[
        Optional[str],
        "Redis connection URL for checkpoint storage (e.g., 'redis://redis:6379'). "
        "If not provided, uses in-memory storage for testing.",
    ] = None
    temperature: Annotated[float, "The temperature for the model"] = 0.7
    max_tokens: Annotated[int, "The maximum tokens for responses"] = 150

    @field_validator("api_key", mode="before")
    @classmethod
    def convert_api_key_to_secret_str(cls, value):
        """Convert string api_key to SecretStr."""
        if isinstance(value, str):
            value = SecretStr(value)
        return value


@serve.deployment
class OnticalModelServer(Generic[SCHEMA]):
    """
    A Ray server that serves an Ontical model.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        schema: Type[SCHEMA],
        temperature: float = 0.7,
        max_tokens: int = 150,
        redis_url: Optional[str] = None,
    ):
        """
        Create a Ray deployment of a LangGraphAgent.

        :param model_name: The name of the LLM model to use
        :param base_url: The base URL for the OpenAI-compatible API
        :param api_key: The API key (can be a dummy value for Ollama)
        :param schema: The Pydantic schema for structured output
        :param temperature: The temperature for the model
        :param max_tokens: The maximum tokens for responses
        :param redis_url: Optional Redis URL for persistent checkpointing
        """
        llm_model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Configure checkpointer based on redis_url
        checkpointer: Optional[BaseCheckpointSaver] = None
        if redis_url:
            from loguru import logger as log

            log.debug(f"Connecting to Redis at {redis_url}")
            checkpointer = RedisSaver.from_conn_string(redis_url)
            log.debug("Setting up Redis indices...")
            checkpointer.setup()  # Create required indices
            log.info("Redis checkpointer initialized successfully")

        self.model = LangGraphAgent(
            llm_model, schema, initial_prompt=None, checkpointer=checkpointer
        )
        self.thread_prompts: dict[str, str] = {}

    async def __call__(self, request: Request) -> tuple[str, SCHEMA]:
        """
        Handle an HTTP request and return a response.

        Expects JSON body with [thread_id, content] format.

        :return: The LLM's response containing optional speech and/or thought
        """
        args = await request.json()
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("Expected JSON array with [thread_id, content]")
        thread_id, content = args
        # Set the initial prompt from thread_prompts if configured
        if thread_id in self.thread_prompts:
            self.model.initial_prompt = self.thread_prompts[thread_id]
        else:
            self.model.initial_prompt = None
        return self.model(thread_id, content)

    def set_thread_prompt(self, thread_id: str, prompt: str) -> None:
        """
        Set or update the initial prompt for a specific thread.

        :param thread_id: The thread identifier
        :param prompt: The initial prompt to use for this thread
        """
        self.thread_prompts[thread_id] = prompt

    def get_thread_prompt(self, thread_id: str) -> Optional[str]:
        """
        Get the initial prompt for a specific thread.

        :param thread_id: The thread identifier
        :return: The thread-specific prompt if set, otherwise None
        """
        return self.thread_prompts.get(thread_id)

    def delete_thread_prompt(self, thread_id: str) -> None:
        """
        Remove the custom prompt for a thread, falling back to the default prompt.

        :param thread_id: The thread identifier
        """
        self.thread_prompts.pop(thread_id, None)

    def list_thread_prompts(self) -> dict[str, str]:
        """
        List all custom thread prompts.

        :return: A copy of the thread prompts mapping
        """
        return self.thread_prompts.copy()

    def clear_thread_checkpoint(self, thread_id: str) -> None:
        """
        Clear the checkpoint data for a specific thread.

        This removes all conversation history stored in the MemorySaver for the
        given thread, freeing up memory. Should be called when a thread is no
        longer needed to prevent memory leaks.

        :param thread_id: The thread identifier
        """
        self.model.clear_thread_checkpoint(thread_id)


def app_builder(args: OnticalModelServerArgs) -> serve.Application:
    """
    Builder function for OnticalModelServer deployment.

    This function enables the Ray Serve builder pattern, allowing configuration
    to be passed via CLI arguments or YAML config files without modifying code.

    :param args: Configuration arguments for the deployment
    :return: A Ray Serve application ready for deployment

    Example usage via CLI:
        serve run server:app_builder model_name="llama3.2:1b" \\
            base_url="http://localhost:11434/v1" \\
            schema_class="test.schemas.Colors"

    Example usage via YAML config:
        applications:
          - name: ontical-model-server
            import_path: ontical_model.server:app_builder
            args:
              model_name: "llama3.2:1b"
              base_url: "http://localhost:11434/v1"
              schema_class: "test.schemas.Colors"
              schema_path: "/path/to/schemas"  # Optional
    """
    # Add schema_path to sys.path if provided
    if args.schema_path:
        sys.path.insert(0, args.schema_path)

    # Dynamically import the schema class
    module_name, class_name = args.schema_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    schema_class: Type[BaseModel] = getattr(module, class_name)

    # Bind the deployment with configuration from args
    return OnticalModelServer.bind(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        schema=schema_class,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        redis_url=args.redis_url,
    )
