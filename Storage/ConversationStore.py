from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Optional
from datetime import datetime, timezone

Role = Literal["user", "assistant", "system"]
HistoryType = List[Tuple[Role, str]]

@dataclass
class Conversation:
    conversation_id: str
    history: HistoryType = field(default_factory=list)
    initial_query: Optional[str] = None
    feedback_pairs: List[Dict[str, Any]] = field(default_factory=list)
    pending_suggestions: List[Dict[str, Any]] = field(default_factory=list)

class ConversationStore:
    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    @staticmethod
    def _compact_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        compact = []
        for item in suggestions or []:
            sug = ""
            if isinstance(item, dict):
                sug = str(item.get("sug", "")).strip()
            elif item is not None:
                sug = str(item).strip()

            if sug:
                compact.append({"sug": sug})

        return compact

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

    def set_initial_query(self, conversation_id: str, query: str) -> Conversation:
        convo = self._require(conversation_id)
        if convo.initial_query is None:
            convo.initial_query = query
        return convo

    def append_feedback_pair(self, conversation_id: str, answer: str) -> Optional[Dict[str, Any]]:
        convo = self._require(conversation_id)
        if not convo.pending_suggestions:
            return None

        pair = {
            "turn": len(convo.feedback_pairs) + 1,
            "suggestions": list(convo.pending_suggestions),
            "answer": answer,
        }
        convo.feedback_pairs.append(pair)
        convo.pending_suggestions = []
        return pair

    def set_pending_suggestions(
        self,
        conversation_id: str,
        suggestions: List[Dict[str, Any]],
    ) -> Conversation:
        convo = self._require(conversation_id)
        convo.pending_suggestions = self._compact_suggestions(suggestions)
        return convo

    def _require(self, conversation_id: str) -> Conversation:
        convo = self.get(conversation_id)
        if convo is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return convo
    
    def clear(self, conversation_id: str) -> Conversation:
        convo = self._require(conversation_id)
        convo.history.clear()
        convo.initial_query = None
        convo.feedback_pairs.clear()
        convo.pending_suggestions.clear()
        return convo

conversation_store = ConversationStore()
