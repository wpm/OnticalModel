from pathlib import Path

from langchain_openai import ChatOpenAI
from ray.serve.handle import DeploymentHandle

from ontical_model.chat_agent import ChatAgent
from ontical_model.schemas import Colors


def test_opposites_model_answer(chat_model: ChatOpenAI, opposites_yml: Path):
    agent = ChatAgent(chat_model, Colors, "Answer questions accurately and succinctly.")
    reply, colors = agent("A", "What are the colors in the American flag?")
    assert reply
    assert colors


def test_opposites_model_answer_via_server(ontical_model_server: DeploymentHandle):
    """
    Test the same functionality as test_opposites_model_answer but using
    OnticalModelServer to host the model.
    """
    # Call the server handle with the same inputs
    reply, colors = ontical_model_server.remote(
        "B", "What are the colors in the American flag?"
    ).result()
    assert reply
    assert colors
    assert isinstance(colors, Colors)
    # Verify that colors were extracted
    assert len(colors.colors) > 0
