from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Optional
from datetime import datetime, timezone

Role = Literal["user", "assistant", "system"]
HistoryType = List[Tuple[Role, str]]

@dataclass
class Conversation:
    conversation_id: str
    history: HistoryType = field(default_factory=list)

class ConversationStore:
    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    def create(self, conversation_id: str) -> Conversation:
        """Create an empty conversation (no media yet)."""
        """In-memory conversation store for text-only multi-turn chat."""
        convo = Conversation(conversation_id=conversation_id)
        self._store[conversation_id] = convo
        return convo

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._store.get(conversation_id)

    def append_message(self, conversation_id: str, role: Role, content: str) -> Conversation:
        convo = self._require(conversation_id)
        convo.history.append((role, content))
        return convo

    def _require(self, conversation_id: str) -> Conversation:
        convo = self.get(conversation_id)
        if convo is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return convo
    
    def clear(self, conversation_id: str) -> Conversation:
        convo = self._require(conversation_id)
        convo.history.clear()
        return convo

conversation_store = ConversationStore()