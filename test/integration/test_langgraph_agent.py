from langchain_openai import ChatOpenAI

from ontical_model.langgraph_agent import LangGraphAgent
from test.schemas import Colors


def test_flag_color_model_call(colors_model: ChatOpenAI):
    agent = LangGraphAgent(
        colors_model, Colors, "Answer questions accurately and succinctly."
    )
    text_response, structured_response = agent(
        "B", "What are the colors in the American flag?"
    )
    assert text_response
    assert structured_response.colors
