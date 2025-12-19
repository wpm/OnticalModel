"""
Support for pure-chat applications.
"""

from __future__ import annotations

from typing import Optional, Annotated

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """
    An entity's response to a chat message.
    This can specify both a reply and changes to an entity's internal state.
    """

    reply: Annotated[Optional[Reply], "The reply sent to other entities"]
    system_message: [
        Optional[str],
        "An internal system message prompt visible to only this entity",
    ]
    stop: Annotated[bool, "Should the entity leave the chat?"]


class Reply(BaseModel):
    """
    A chat response sent to other entities.
    """

    text: Annotated[str, "The response"]
    recipients: Annotated[
        set[str],
        "IDs of the entities to which the response is sent. Empty set to broadcast to all",
    ] = set()
