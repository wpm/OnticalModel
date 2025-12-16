from pathlib import Path
from typing import Annotated

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ontical_model.chat_agent import ChatAgent


class Colors(BaseModel):
    colors: Annotated[set[str], "All the colors mentioned in the reply"]


def test_opposites_model_answer(chat_model: ChatOpenAI, opposites_yml: Path):
    agent = ChatAgent(chat_model, Colors, "Answer questions accurately and succinctly.")
    reply, colors = agent("A", "What are the colors in the American flag?")
    assert reply
    assert colors
