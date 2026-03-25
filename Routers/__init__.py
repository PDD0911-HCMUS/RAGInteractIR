from fastapi import APIRouter
from Routers.VLMChatRouter import VLMChatRouter
from Storage.ConversationStore import conversation_store


def build_api_router() -> APIRouter:
    api_router = APIRouter()
    vlm = VLMChatRouter(conversation_store=conversation_store)
    api_router.include_router(vlm.router)
    return api_router