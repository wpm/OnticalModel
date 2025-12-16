from typing import Type, Generic, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from ray import serve

from ontical_model.chat_agent import ChatAgent

SCHEMA = TypeVar("SCHEMA", bound=BaseModel)


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
        Create a Ray deployment of a ChatAgent.

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
        self.model = ChatAgent(llm_model, schema, initial_prompt)

    def __call__(self, thread_id: str, content: str) -> tuple[str, SCHEMA]:
        """
        Handle a request from an entity and return a response.

        :return: The LLM's response containing optional speech and/or thought
        """
        return self.model(thread_id, content)
