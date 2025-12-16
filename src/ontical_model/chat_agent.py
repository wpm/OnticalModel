from datetime import datetime
from typing import Type, TypedDict, Annotated, Generic, TypeVar

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel

SCHEMA = TypeVar("SCHEMA", bound=BaseModel)


class ChatAgent(Generic[SCHEMA]):
    def __init__(self, model: BaseChatModel, schema: Type[SCHEMA], initial_prompt: str):
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            structured_response: NotRequired[SCHEMA]

        def llm(state: State) -> dict:
            text_response = self.model.invoke(state["messages"])
            structured_model = self.model.with_structured_output(self.schema)
            structured_response = structured_model.invoke(state["messages"])
            return {
                "messages": [text_response],
                "structured_response": structured_response,
            }

        self.model = model
        self.schema = schema
        self.initial_prompt = initial_prompt
        llm_node = "llm"
        graph_builder = StateGraph(State)
        graph_builder.add_node(llm_node, llm)
        graph_builder.set_entry_point(llm_node)
        graph_builder.add_edge(llm_node, END)
        self.graph = graph_builder.compile(checkpointer=MemorySaver())

    def __call__(self, thread_id: str, content: str) -> tuple[str, SCHEMA]:
        def is_new_thread() -> bool:
            state = self.graph.checkpointer.get(config)
            return state is None or not state.get("channel_values", {}).get("messages")

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        messages = []
        if is_new_thread():
            messages.append(SystemMessage(content=self.initial_prompt))
        messages.append(HumanMessage(content=content))
        agent_input = {"messages": messages}
        result = self.graph.invoke(agent_input, config)
        text_response = result["messages"][-1].content
        structured_response = result["structured_response"]
        return text_response, structured_response


@tool
def time_difference(start: datetime, end: datetime) -> float:
    """
    Calculate the time difference in seconds between two timestamps.

    :param start: The starting timestamp
    :param end: The ending timestamp
    :return: The difference in seconds (positive if end is after start)
    """
    delta = end - start
    return delta.total_seconds()
