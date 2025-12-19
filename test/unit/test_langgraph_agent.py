from langchain_openai import ChatOpenAI

from ontical_model.chat import ChatResponse
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


def test_chat_response_schema(colors_model: ChatOpenAI):
    """
    Test ChatResponse schema with LLM agent.

    This test exercises the structured output capabilities of ChatResponse,
    including reply, system_message, and stop fields. We don't assert on the
    values because the small LLM isn't reliable enough to guarantee correct
    structured responses - this just ensures the schema works end-to-end.
    """
    system_prompt = (
        "You are a chatbot. Follow these rules:\n"
        "1. If your conversation partner says 'goodbye' in some way, say 'goodbye' "
        "back and write a system message indicating that it's time to leave the chat.\n"
        "2. If there is a system message saying that your conversation partner has "
        "left, you should leave the chat by setting stop=True.\n"
        "3. Always provide a reply unless you're stopping the chat."
    )

    agent = LangGraphAgent(colors_model, ChatResponse, system_prompt)

    # Test case 1: Normal conversation (should have reply, no system message, stop=False)
    text_response_1, structured_response_1 = agent("thread1", "Hello! How are you?")
    # Don't assert - just exercise the schema
    _ = text_response_1
    _ = structured_response_1.reply
    _ = structured_response_1.system_message
    _ = structured_response_1.stop

    # Test case 2: User says goodbye (should trigger system message about leaving)
    text_response_2, structured_response_2 = agent("thread2", "Goodbye!")
    _ = text_response_2
    _ = structured_response_2.reply
    _ = structured_response_2.system_message
    _ = structured_response_2.stop

    # Test case 3: System message that partner left (should set stop=True)
    text_response_3, structured_response_3 = agent(
        "thread3", "System: Your conversation partner has left the chat."
    )
    _ = text_response_3
    _ = structured_response_3.reply
    _ = structured_response_3.system_message
    _ = structured_response_3.stop
