import os

from fastapi import APIRouter
from Routers.RAIRRouter import RAIRRouter
from Routers.VLMChatRouter import VLMChatRouter
from Services.RAIRInteractiveRetrievalService import RAIRInteractiveRetrievalService
from Storage.ConversationStore import conversation_store


vlm_chat_router = VLMChatRouter(conversation_store=conversation_store)
rair_router = RAIRRouter(service=RAIRInteractiveRetrievalService())


def build_api_router() -> APIRouter:
    api_router = APIRouter()
    api_router.include_router(vlm_chat_router.router)
    api_router.include_router(rair_router.router)
    return api_router


def preload_gallery() -> None:
    if os.environ.get("PRELOAD_LEGACY_VLM", "0") == "1":
        vlm_chat_router.preload_gallery()
    if os.environ.get("PRELOAD_RAIR_API", "1") == "1":
        rair_router.preload_gallery()
