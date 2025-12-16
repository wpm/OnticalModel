import requests
from langchain_openai import ChatOpenAI

from ontical_model.chat_agent import ChatAgent
from ontical_model.test_schemas import Colors


def test_opposites_model_answer(chat_model: ChatOpenAI):
    agent = ChatAgent(chat_model, Colors, "Answer questions accurately and succinctly.")
    reply, colors = agent("A", "What are the colors in the American flag?")
    assert reply
    assert colors


def test_opposites_model_answer_via_server(ontical_model_server_url: str):
    """
    Test the same functionality as test_opposites_model_answer but using
    OnticalModelServer deployed via docker-compose and accessed over HTTP.
    """
    # Call the deployed server via HTTP
    # Ray Serve expects JSON body with positional args
    response = requests.post(
        ontical_model_server_url,
        json=["B", "What are the colors in the American flag?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"

    # Parse the response - Ray Serve returns the tuple as JSON array
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 2

    reply = result[0]
    colors_dict = result[1]

    assert reply
    assert colors_dict
    assert "colors" in colors_dict
    assert len(colors_dict["colors"]) > 0
