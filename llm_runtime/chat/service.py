"""Chat orchestration helpers for llm-driven runtime control."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..llm.chat_client import ChatMessage, ChatResponse


class ChatService:
    """Encapsulate chat parsing and llm request construction."""

    # Build a transient chat history including typing placeholder for immediate UI feedback.
    def history_with_typing_placeholder(
        self, chat_history: List[Tuple[str, str]], message: str
    ) -> List[Tuple[str, str]]:
        """Return rendered history where the pending assistant bubble shows typing."""
        return list(chat_history) + [("user", message), ("assistant", "typing...")]

    # Build outbound message objects for llm clients from internal history tuples.
    def build_outbound_messages(self, chat_history: List[Tuple[str, str]]) -> List[ChatMessage]:
        """Convert (role,text) tuples to ChatMessage list for client.send()."""
        return [ChatMessage(role=r, content=t) for r, t in chat_history]

    # Invoke llm client if available, otherwise return default queued-command response.
    def send_to_llm(
        self, chat_client, chat_history: List[Tuple[str, str]], sim_context: Dict
    ) -> ChatResponse:
        """Return ChatResponse from active backend or a fallback local response."""
        if chat_client is None:
            return ChatResponse(text="Command queued.")
        return chat_client.send(self.build_outbound_messages(chat_history), sim_context)
