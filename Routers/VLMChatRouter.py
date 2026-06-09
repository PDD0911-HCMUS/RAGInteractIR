import os
import uuid
import time
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Services.VisDialGPTCLIPService import VisDialGPTCLIPService
from Services.OpenAIService import OpenAIService
from Services.PromptCollectionService import PromptCollectionService
from Storage.ConversationStore import ConversationStore


logger = logging.getLogger("rair")


def resolve_torch_device(env_name: str, default: str = "cpu") -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return default


class MessageRequest(BaseModel):
    message: str


class VLMChatRouter:
    def __init__(
        self,
        conversation_store: ConversationStore,
        upload_dir: Optional[str] = None,
    ):
        # -------------- Config --------------- #
        self.store = conversation_store
        self.upload_dir = upload_dir or os.environ.get("VLM_UPLOAD_DIR", "./uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        # Configurable runtime
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_device = resolve_torch_device("CLIP_DEVICE")
        self.greeting_model = "gpt-5.4"
        self.reasoning_model = "gpt-5.4"

        self.prompt = PromptCollectionService()
        self.openai_service: Optional[OpenAIService] = None
        self.visdial_serve: Optional[VisDialGPTCLIPService] = None
        self.gallery = None

        # ---------------- API ---------------- #
        self.router = APIRouter(prefix="/vlm", tags=["VLM"])

        self.router.add_api_route(
            "/conversations",
            self.create_conversation,
            methods=["GET"],
            summary="Create a new conversation (no media yet)",
        )

        self.router.add_api_route(
            "/conversations/{conversation_id}/messages",
            self.send_message,
            methods=["POST"],
            summary="Send a question to existing conversation",
        )

        self.router.add_api_route(
            "/conversations/{conversation_id}",
            self.get_conversation,
            methods=["GET"],
            summary="Get conversation state (debug)",
        )

    def _get_openai_service(self) -> OpenAIService:
        if self.openai_service is None:
            self.openai_service = OpenAIService(model_name=self.greeting_model)
        return self.openai_service

    def _get_visdial_service(self) -> VisDialGPTCLIPService:
        if self.visdial_serve is None:
            self.visdial_serve = VisDialGPTCLIPService(
                vlm=self.clip_model_name,
                device=self.clip_device,
                openai_service=self._get_openai_service(),
                reasoning_model=self.reasoning_model,
            )
        return self.visdial_serve

    def _get_gallery(self):
        if self.gallery is None:
            self.gallery = self._get_visdial_service().build_gallery()
        return self.gallery

    def _build_context_state(self, convo, latest_user_message: Optional[str] = None):
        return {
            "initial_query": getattr(convo, "initial_query", None),
            "feedback_pairs": getattr(convo, "feedback_pairs", []),
            "pending_suggestions": getattr(convo, "pending_suggestions", []),
            "latest_user_message": latest_user_message,
        }

    def preload_gallery(self) -> None:
        if os.environ.get("PRELOAD_CLIP_MODEL", "0") == "1":
            logger.info("Preloading CLIP model")
            self._get_visdial_service().preload_model()

        logger.info("Preloading VisDial gallery")
        self._get_gallery()

        logger.info("VisDial gallery ready")

    async def create_conversation(self):
        """
        GET /api/v1/vlm/conversations
        Create an empty conversation.
        """
        conversation_id = str(uuid.uuid4())
        self.store.create(conversation_id=conversation_id)

        # Seed system prompt
        self.store.append_message(
            conversation_id,
            "system",
            "You are a helpful vision-language assistant."
        )

        # Greeting via OpenAI, not local Qwen
        start = time.time()
        caption = self._get_openai_service().generate_answer(
            user_prompt=self.prompt.greeting,
            history=None,
            model=self.greeting_model,
            temperature=0.2,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        self.store.append_message(conversation_id, "assistant", caption)

        return {
            "conversation_id": conversation_id,
            "caption": caption,
            "meta": {
                "backend": "openai",
                "model": self.greeting_model,
                "latency_ms": latency_ms,
            },
        }

    async def get_conversation(self, conversation_id: str):
        """
        GET /api/v1/vlm/conversations/{conversation_id}
        Debug endpoint.
        """
        convo = self.store.get(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "conversation_id": convo.conversation_id,
            "history": getattr(convo, "history", []),
            "initial_query": getattr(convo, "initial_query", None),
            "feedback_pairs": getattr(convo, "feedback_pairs", []),
            "pending_suggestions": getattr(convo, "pending_suggestions", []),
            "context_state": self._build_context_state(convo),
        }

    async def send_message(self, conversation_id: str, req: MessageRequest):
        """
        POST /api/v1/vlm/conversations/{conversation_id}/messages
        JSON:
          - message: str
        """
        convo = self.store.get(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        user_message = (req.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message must not be empty")

        if convo.initial_query is None:
            self.store.set_initial_query(conversation_id, user_message)
        else:
            self.store.append_feedback_pair(conversation_id, user_message)

        context_state = self._build_context_state(convo, latest_user_message=user_message)
        logger.info(
            "RAIR turn: conversation=%s feedback_pairs=%d user_message=%s",
            conversation_id,
            len(context_state["feedback_pairs"]),
            user_message,
        )
        logger.info(
            "Context State C_t:\n%s",
            json.dumps(context_state, ensure_ascii=False, indent=2),
        )

        # Step 1: Rewrite C_t into a natural-language retrieval query
        try:
            visdial_serve = self._get_visdial_service()
            rewritten_query, rewrite_latency_ms = await visdial_serve.rewrite_query(
                context_state=context_state,
            )
            logger.info(
                "Rewritten query: %s latency_ms=%d",
                rewritten_query,
                rewrite_latency_ms,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Context rewrite failed: {repr(e)}"
            ) from e

        # Step 2: Append user message to conversation history
        self.store.append_message(conversation_id, "user", user_message)

        # For reasoning/retrieval, use history that includes the new user turn
        convo_after_user = self.store.get(conversation_id)
        history_for_reasoning = list(convo_after_user.history)

        # Step 3: Retrieval with text query + candidate-grounded reasoning
        try:
            retrieve = await visdial_serve.RAG_faiss_retrieval(
                history_for_reasoning,
                self._get_gallery(),
                rewritten_query,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"RAG retrieval/reasoning failed: {repr(e)}"
            ) from e

        # Step 4: Append assistant payload
        assistant_payload = json.dumps(
            {
                "rewritten_query": rewritten_query,
                "diagnosis": retrieve.get("diagnosis", {}),
                "suggestions": retrieve.get("suggest", []),
            },
            ensure_ascii=False,
        )
        self.store.append_message(conversation_id, "assistant", assistant_payload)
        self.store.set_pending_suggestions(
            conversation_id,
            retrieve.get("suggest", []),
        )

        # Re-fetch to get the latest length accurately
        convo_final = self.store.get(conversation_id)
        trace = {
            "context_state": self._build_context_state(
                convo_final,
                latest_user_message=user_message,
            ),
            "rewritten_query": rewritten_query,
            "candidate_evidence": retrieve.get("candidate_evidence", [])[:8],
            "diagnosis": retrieve.get("diagnosis", {}),
            "top5": [
                {
                    "rank": rank,
                    "image_id": image_id,
                    "score": score,
                    "caption": caption,
                }
                for rank, (image_id, score, caption) in enumerate(
                    zip(
                        retrieve.get("id", [])[:5],
                        retrieve.get("score", [])[:5],
                        retrieve.get("text", [])[:5],
                    ),
                    start=1,
                )
            ],
            "suggestions": retrieve.get("suggest", []),
        }
        logger.info(
            "RAIR turn complete: conversation=%s rewrite_ms=%d suggestions=%d",
            conversation_id,
            rewrite_latency_ms,
            len(retrieve.get("suggest", [])),
        )

        return {
            "conversation_id": conversation_id,
            "answer": rewritten_query,
            "rewritten_query": rewritten_query,
            "diagnosis": retrieve.get("diagnosis", {}),
            "history_length": len(convo_final.history) if convo_final else 0,
            "initial_query": getattr(convo_final, "initial_query", None),
            "feedback_pairs": getattr(convo_final, "feedback_pairs", []),
            "pending_suggestions": getattr(convo_final, "pending_suggestions", []),
            "context_state": self._build_context_state(
                convo_final,
                latest_user_message=user_message,
            ),
            "trace": trace,
            "retrieve": retrieve,
            "meta": {
                "rewrite_latency_ms": rewrite_latency_ms,
                "rewrite_model": self.reasoning_model,
                "reasoning_model": self.reasoning_model,
            },
        }
