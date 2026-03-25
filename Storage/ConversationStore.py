from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Optional
from datetime import datetime, timezone

Role = Literal["user", "assistant", "system"]
HistoryType = List[Tuple[Role, str]]
MediaType = Literal["image", "video", "audio", "file", "unknown"]

@dataclass
class MediaItem:
    media_path: str
    media_type: MediaType = "unknown"
    uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class Conversation:
    conversation_id: str
    history: HistoryType = field(default_factory=list)

    # current media (useful when your pipeline assumes 1 active media at a time)
    media_path: Optional[str] = None
    media_type: MediaType = "unknown"

    # keep all uploaded media for future extension
    media_items: List[MediaItem] = field(default_factory=list)


class ConversationStore:
    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    def create(self, conversation_id: str) -> Conversation:
        """Create an empty conversation (no media yet)."""
        convo = Conversation(conversation_id=conversation_id)
        self._store[conversation_id] = convo
        return convo

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._store.get(conversation_id)

    def append_message(self, conversation_id: str, role: Role, content: str) -> Conversation:
        convo = self._require(conversation_id)
        convo.history.append((role, content))
        return convo

    def attach_media(self, conversation_id: str, media_path: str, media_type: MediaType = "unknown") -> Conversation:
        """
        Attach/upload media AFTER conversation already exists.
        - Updates the active media (media_path/media_type)
        - Also stores it in media_items for later use (multi-media support)
        """
        convo = self._require(conversation_id)
        item = MediaItem(media_path=media_path, media_type=media_type)
        convo.media_items.append(item)

        # set current/active media
        convo.media_path = media_path
        convo.media_type = media_type

        # optional: log it into history so downstream can reconstruct state from history alone
        convo.history.append(("system", f"[media_attached] type={media_type} path={media_path}"))
        return convo

    def _require(self, conversation_id: str) -> Conversation:
        convo = self.get(conversation_id)
        if convo is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return convo


conversation_store = ConversationStore()