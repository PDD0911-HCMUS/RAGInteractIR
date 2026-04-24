import os
import uuid
import time
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Services.VisDialGPTCLIPService import VisDialGPTCLIPService
from Services.OpenAIService import OpenAIService
from Services.PromptCollectionService import PromptCollectionService
from Storage.ConversationStore import ConversationStore


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
        self.clip_device = "cuda"
        self.greeting_model = "gpt-5.4"
        self.triplet_model = "gpt-5.4"
        self.reasoning_model = "gpt-5.4"

        # Shared OpenAI service
        self.openai_service = OpenAIService(model_name=self.greeting_model)

        # Retrieval service
        self.visdial_serve = VisDialGPTCLIPService(
            vlm=self.clip_model_name,
            device=self.clip_device,
            openai_service=self.openai_service,
            triplet_model=self.triplet_model,
            reasoning_model=self.reasoning_model,
        )

        self.prompt = PromptCollectionService()
        self.gallery = self.visdial_serve.build_gallery()

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
        caption = self.openai_service.generate_answer(
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

        print(f"User Message: {user_message}")

        # Snapshot current history BEFORE modifying it
        history_before = list(convo.history)

        # Step 1: Convert to triplets
        try:
            triplets, latency_ms = await self.visdial_serve.convert_triplet(
                text=user_message,
                history=history_before,
            )
            print(f"History Before: {history_before}")
            print(f"Triplet Answer: {triplets}")

            triplet_json = json.loads(triplets)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Triplet response is not valid JSON: {triplets}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Triplet conversion failed: {repr(e)}"
            ) from e

        # Step 2: Build retrieval query
        try:
            queries = await self.visdial_serve.build_query(answer=triplet_json)
            print(f"Queries Built: {queries}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build query from triplets: {repr(e)}"
            ) from e

        # Step 3: Append user message to conversation history
        self.store.append_message(conversation_id, "user", user_message)

        # For reasoning/retrieval, use history that includes the new user turn
        convo_after_user = self.store.get(conversation_id)
        history_for_reasoning = list(convo_after_user.history)

        # Step 4: Retrieval + reasoning
        try:
            retrieve = await self.visdial_serve.RAG_faiss_retrieval(
                history_for_reasoning,
                self.gallery,
                queries,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"RAG retrieval/reasoning failed: {repr(e)}"
            ) from e

        # Step 5: Append assistant payload
        assistant_payload = json.dumps(
            {
                "triplets": triplets,
                "suggestions": retrieve.get("suggest", []),
            },
            ensure_ascii=False,
        )
        self.store.append_message(conversation_id, "assistant", assistant_payload)

        # Re-fetch to get the latest length accurately
        convo_final = self.store.get(conversation_id)

        return {
            "conversation_id": conversation_id,
            "answer": triplets,
            "history_length": len(convo_final.history) if convo_final else 0,
            "retrieve": retrieve,
            "meta": {
                "latency_ms": latency_ms,
                "triplet_model": self.triplet_model,
                "reasoning_model": self.reasoning_model,
            },
        }