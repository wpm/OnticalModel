import importlib
import sys
from typing import Type, Generic, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, Field
from ray import serve
from starlette.requests import Request

from ontical_model.langgraph_agent import LangGraphAgent

SCHEMA = TypeVar("SCHEMA", bound=BaseModel)


class OnticalModelServerArgs(BaseModel):
    """
    Configuration arguments for OnticalModelServer deployment.
    """

    model_name: str = Field(
        description="The name of the LLM model to use (e.g., 'llama3.2:1b')"
    )
    base_url: str = Field(description="The base URL for the OpenAI-compatible API")
    api_key: str = Field(
        default="ollama", description="The API key (can be a dummy value for Ollama)"
    )
    schema_class: str = Field(
        description="Fully qualified name of the Pydantic schema class "
        "(e.g., 'test.schemas.Colors')"
    )
    schema_path: str | None = Field(
        default=None,
        description="Optional path to directory containing schema module. "
        "If provided, this path will be added to sys.path before importing schema_class.",
    )
    initial_prompt: str = Field(
        default="Answer questions accurately and succinctly.",
        description="The initial system prompt",
    )
    temperature: float = Field(default=0.7, description="The temperature for the model")
    max_tokens: int = Field(default=150, description="The maximum tokens for responses")


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
        initial_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ):
        """
        Create a Ray deployment of a LangGraphAgent.

        :param model_name: The name of the LLM model to use
        :param base_url: The base URL for the OpenAI-compatible API
        :param api_key: The API key (can be a dummy value for Ollama)
        :param schema: The Pydantic schema for structured output
        :param initial_prompt: The initial system prompt
        :param temperature: The temperature for the model
        :param max_tokens: The maximum tokens for responses
        """
        llm_model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model = LangGraphAgent(llm_model, schema, initial_prompt)

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
        return self.model(thread_id, content)


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
        initial_prompt=args.initial_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
