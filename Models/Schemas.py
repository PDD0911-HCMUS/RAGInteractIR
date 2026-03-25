from pydantic import BaseModel
from typing import List, Optional

class TextContent(BaseModel):
    type: str = "text"
    text: str

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: List[TextContent]

class ImageInput(BaseModel):
    type: str  # "base64"
    data: str
    mime: str

class ChatOptions(BaseModel):
    language: Optional[str] = "en"
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    num_captions: Optional[int] = 1

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    image: Optional[ImageInput] = None
    message: Message
    options: Optional[ChatOptions] = ChatOptions()

class ChatResponse(BaseModel):
    conversation_id: str
    message: Message
    meta: dict
