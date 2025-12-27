"""Unit tests for chat message models."""

import pytest
from pydantic import ValidationError

from ontical_model.chat import ChatResponse, Reply


def test_reply_with_recipients():
    """Test Reply model with specific recipients."""
    reply = Reply(text="Hello, world!", recipients={"user1", "user2"})
    assert reply.text == "Hello, world!"
    assert reply.recipients == {"user1", "user2"}


def test_reply_broadcast():
    """Test Reply model with empty recipients (broadcast)."""
    reply = Reply(text="Hello, everyone!")
    assert reply.text == "Hello, everyone!"
    assert reply.recipients == set()


def test_reply_text_required():
    """Test that Reply requires text field."""
    with pytest.raises(ValidationError) as exc_info:
        Reply(recipients={"user1"})
    assert "text" in str(exc_info.value)


def test_chat_response_with_reply():
    """Test ChatResponse with a reply."""
    reply = Reply(text="Hello!", recipients={"user1"})
    response = ChatResponse(reply=reply, system_message=None, stop=False)
    assert response.reply == reply
    assert response.system_message is None
    assert response.stop is False


def test_chat_response_with_system_message():
    """Test ChatResponse with a system message."""
    response = ChatResponse(
        reply=None, system_message="Internal note: user is leaving", stop=False
    )
    assert response.reply is None
    assert response.system_message == "Internal note: user is leaving"
    assert response.stop is False


def test_chat_response_stop():
    """Test ChatResponse with stop flag set."""
    response = ChatResponse(reply=None, system_message=None, stop=True)
    assert response.reply is None
    assert response.system_message is None
    assert response.stop is True


def test_chat_response_all_fields():
    """Test ChatResponse with all fields populated."""
    reply = Reply(text="Goodbye!", recipients=set())
    response = ChatResponse(
        reply=reply, system_message="Conversation ending", stop=True
    )
    assert response.reply == reply
    assert response.system_message == "Conversation ending"
    assert response.stop is True


def test_chat_response_stop_required():
    """Test that ChatResponse requires stop field."""
    with pytest.raises(ValidationError) as exc_info:
        ChatResponse(reply=None, system_message=None)
    assert "stop" in str(exc_info.value)
