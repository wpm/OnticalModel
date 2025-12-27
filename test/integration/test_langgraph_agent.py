from langchain_openai import ChatOpenAI

from ontical_model.langgraph_agent import LangGraphAgent
from test.schemas import NameAgeOccupation


def test_name_query_direct_call(
    name_age_occupation_model: ChatOpenAI,
    bob_prompt: str,
    bob_thread_ids: list[str],
):
    """Test that the agent can handle a name query."""
    agent = LangGraphAgent(name_age_occupation_model, NameAgeOccupation, bob_prompt)
    text_response, structured_response = agent(bob_thread_ids[0], "What is your name?")
    assert text_response
    assert isinstance(structured_response, NameAgeOccupation)
    assert structured_response.name == "Bob"


def test_age_query_direct_call(
    name_age_occupation_model: ChatOpenAI,
    bob_prompt: str,
    bob_thread_ids: list[str],
):
    """Test that the agent can handle an age query."""
    agent = LangGraphAgent(name_age_occupation_model, NameAgeOccupation, bob_prompt)
    text_response, structured_response = agent(bob_thread_ids[1], "How old are you?")
    assert text_response
    assert isinstance(structured_response, NameAgeOccupation)
    assert structured_response.age == 28


def test_occupation_query_direct_call(
    name_age_occupation_model: ChatOpenAI,
    bob_prompt: str,
    bob_thread_ids: list[str],
):
    """Test that the agent can handle an occupation query."""
    agent = LangGraphAgent(name_age_occupation_model, NameAgeOccupation, bob_prompt)
    text_response, structured_response = agent(
        bob_thread_ids[2], "What is your occupation?"
    )
    assert text_response
    assert isinstance(structured_response, NameAgeOccupation)
    assert structured_response.occupation == "bartender"
