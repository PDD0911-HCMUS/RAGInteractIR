from fastapi import APIRouter
from Routers.VLMChatRouter import VLMChatRouter
from Storage.ConversationStore import conversation_store


vlm_chat_router = VLMChatRouter(conversation_store=conversation_store)


def build_api_router() -> APIRouter:
    api_router = APIRouter()
    api_router.include_router(vlm_chat_router.router)
    return api_router


def preload_gallery() -> None:
    vlm_chat_router.preload_gallery()
