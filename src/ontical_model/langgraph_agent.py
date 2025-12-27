from typing import Type, TypedDict, Annotated, Generic, TypeVar, Optional

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel

SCHEMA = TypeVar("SCHEMA", bound=BaseModel)


class LangGraphAgent(Generic[SCHEMA]):
    """
    LangGraphAgent is a wrapper around a single-node LangGraph agent structured outputs.
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: Type[SCHEMA],
        initial_prompt: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        :param model: The language model used for invoking and generating responses.
        :param schema: The schema type that defines the structured output for the model.
        :param initial_prompt: Optional initial system message to prepend to a new thread
        :param checkpointer: Optional checkpoint saver (defaults to MemorySaver for testing)
        """

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            # noinspection PyTypeHints
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
        # noinspection PyTypeChecker
        graph_builder = StateGraph(State)
        # noinspection PyTypeChecker
        graph_builder.add_node(llm_node, llm)
        graph_builder.set_entry_point(llm_node)
        graph_builder.add_edge(llm_node, END)

        # Use provided checkpointer or default to MemorySaver for testing
        if checkpointer is None:
            checkpointer = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=checkpointer)

    def __call__(self, thread_id: str, content: str) -> tuple[str, SCHEMA]:
        """
        Call the model with the given thread ID and content.

        :param thread_id: Identifier for the conversational thread.
        :param content: Message for the model to reply to.
        :return: A tuple containing the textual response and a structured response object.
        """

        def is_new_thread() -> bool:
            state = self.graph.checkpointer.get(config)
            return state is None or not state.get("channel_values", {}).get("messages")

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        messages = []
        if is_new_thread() and self.initial_prompt:
            messages.append(SystemMessage(content=self.initial_prompt))
        messages.append(HumanMessage(content=content))
        agent_input = {"messages": messages}
        # noinspection PyTypeChecker
        result = self.graph.invoke(agent_input, config)
        text_response = result["messages"][-1].content
        structured_response = result["structured_response"]
        return text_response, structured_response

    def clear_thread_checkpoint(self, thread_id: str) -> None:
        """
        Clear the checkpoint data for a specific thread.

        This removes all conversation history stored in the checkpointer for the
        given thread, freeing up memory. Should be called when a thread is no
        longer needed to prevent memory leaks.

        :param thread_id: The thread identifier
        """
        self.graph.checkpointer.delete_thread(thread_id)
