from langchain_openai import ChatOpenAI

from ontical_model.langgraph_agent import LangGraphAgent
from test.schemas import NameAgeOccupation


def test_name_age_occupation_query(
    name_age_occupation_model: ChatOpenAI, bob_prompt: str
):
    """Test that the agent can extract name, age, and occupation from conversation."""
    agent = LangGraphAgent(name_age_occupation_model, NameAgeOccupation, bob_prompt)

    # Ask for name
    text_response, structured_response = agent("test_thread", "What is your name?")
    assert text_response
    # Note: We don't assert specific values because small LLMs may not be reliable

    # Ask for age
    text_response, structured_response = agent("test_thread", "How old are you?")
    assert text_response

    # Ask for occupation
    text_response, structured_response = agent(
        "test_thread", "What is your occupation?"
    )
    assert text_response
    assert (
        NameAgeOccupation(name="Bob", age=28, occupation="bartender")
        == structured_response
    )
