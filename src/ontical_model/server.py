from pydantic import BaseModel
from ray import serve

from ontical_model.chat_agent import ChatAgent


@serve.deployment
class OnticalModelServer[SCHEMA: BaseModel]:
    """
    A Ray server that serves an Ontical model.
    """

    def __init__(self, model: ChatAgent):
        """
        Create a Ray deployment of a MultiChatModel.

        :param model: The MultiChatModel to serve
        """
        self.model = model

    def __call__(self, thread_id: str, content: str) -> tuple[str, SCHEMA]:
        """
        Handle a request from an entity and return a response.

        :return: The LLM's response containing optional speech and/or thought
        """
        return self.model(thread_id, content)
