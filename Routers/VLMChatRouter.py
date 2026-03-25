import os
import uuid
import time
from typing import Optional, Protocol, Literal
import json
import faiss
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
from pydantic import BaseModel
import torch

from Services.QwenVLService import generate_caption, generate_answer
# from Services.VisDialSigLIPService import VisDialSiGLIPService
from Services.VisDialCLIPService import VisDialCLIPService


# ---- typing shared ----
Role = Literal["user", "assistant", "system"]
MediaType = Literal["image", "video", "audio", "file", "unknown"]


class MessageRequest(BaseModel):
    message: str


# ---- Storage interface (so you can swap implementations easily) ----
class ConversationProtocol(Protocol):
    conversation_id: str
    history: list[tuple[Role, str]]
    media_path: Optional[str]
    media_type: MediaType


class ConversationStoreProtocol(Protocol):
    def create(self, conversation_id: str) -> ConversationProtocol: ...
    def get(self, conversation_id: str) -> Optional[ConversationProtocol]: ...
    def append_message(self, conversation_id: str, role: Role, content: str) -> ConversationProtocol: ...
    def attach_media(self, conversation_id: str, media_path: str, media_type: MediaType = "unknown") -> ConversationProtocol: ...


class VLMChatRouter:
    def __init__(
        self,
        conversation_store: ConversationStoreProtocol,
        upload_dir: Optional[str] = None,
    ):
        self.store = conversation_store
        self.upload_dir = upload_dir or os.environ.get("VLM_UPLOAD_DIR", "./uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # self.visdial_serve = VisDialSiGLIPService(
        #     device="cuda" if torch.cuda.is_available() else "cpu",
        #     vlm="google/siglip-base-patch16-224"
        # )
        
        self.visdial_serve = VisDialCLIPService(
            device="cuda" if torch.cuda.is_available() else "cpu",
            vlm="openai/clip-vit-base-patch32"
        )

        self.router = APIRouter(prefix="/vlm", tags=["VLM"])
        
        self.gallery = self.visdial_serve.build_gallery()

        # 1) Create empty conversation
        self.router.add_api_route(
            "/conversations",
            self.create_conversation,
            methods=["GET"],
            summary="Create a new conversation (no media yet)",
        )

        # 2) Upload/attach media later (currently image-only)
        self.router.add_api_route(
            "/conversations/{conversation_id}/media",
            self.upload_media,
            methods=["POST"],
            summary="Upload a media file (currently image-only) and attach to conversation",
        )

        # 3) Send message (uses current media if available)
        self.router.add_api_route(
            "/conversations/{conversation_id}/messages",
            self.send_message,
            methods=["POST"],
            summary="Send a question to existing conversation (uses current media if available)",
        )

        # 4) (Optional utility) Get conversation state
        self.router.add_api_route(
            "/conversations/{conversation_id}",
            self.get_conversation,
            methods=["GET"],
            summary="Get conversation state (debug)",
        )
        
        self.router.add_api_route(
            "/conversations/{conversation_id}/faiss",
            self.RAG_faiss_retrieval,
            methods=["POST"],
            summary="faiss search"
        )

    # ----------------- helpers -----------------
    def _validate_image(self, file: UploadFile) -> None:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported (for now).")

    def _media_save_path(self, file: UploadFile, conversation_id: str) -> str:
        """
        Save as <conversation_id>_<uuid8>.<ext> to allow multiple uploads over time.
        """
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
            ext = ".jpg"
        suffix = uuid.uuid4().hex[:8]
        return os.path.join(self.upload_dir, f"{conversation_id}_{suffix}{ext}")

    def _write_bytes(self, path: str, content: bytes) -> None:
        with open(path, "wb") as f:
            f.write(content)

    # ----------------- endpoints -----------------
    async def create_conversation(self):
        """
        POST /api/v1/vlm/conversations
        Create an empty conversation (no media required).
        """
        conversation_id = str(uuid.uuid4())
        self.store.create(conversation_id=conversation_id)

        # Seed system prompt in history (optional but useful)
        self.store.append_message(conversation_id, "system", "You are the Vision-Language Model.")

        # Optional initial assistant greeting (text-only)
        start = time.time()
        greeting_prompt = "Hello, who are you?"
        caption = generate_caption(media_path=None, user_prompt=greeting_prompt, media_type="unknown")
        latency_ms = int((time.time() - start) * 1000)

        self.store.append_message(conversation_id, "assistant", caption)

        return {
            "conversation_id": conversation_id,
            "caption": caption,
            "meta": {
                "backend": "lmdeploy",
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "latency_ms": latency_ms,
            },
        }

    async def get_conversation(self, conversation_id: str):
        """
        GET /api/v1/vlm/conversations/{conversation_id}
        Debug endpoint to inspect current state.
        """
        convo = self.store.get(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "conversation_id": convo.conversation_id,
            "media_path": getattr(convo, "media_path", None),
            "media_type": getattr(convo, "media_type", "unknown"),
            "history": getattr(convo, "history", []),
        }
    
    async def upload_media(
        self,
        conversation_id: str,
        media: UploadFile = File(...),
        prompt: str = Form("Please describe this image."),
    ):
        """
        POST /api/v1/vlm/conversations/{conversation_id}/media
        multipart/form-data:
          - media: file (image only for now)
          - prompt: string (optional)
        """
        convo = self.store.get(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # now: image only
        self._validate_image(media)

        media_path = self._media_save_path(media, conversation_id)
        content = await media.read()
        self._write_bytes(media_path, content)

        # attach as current media
        self.store.attach_media(conversation_id, media_path=media_path, media_type="image")

        # caption step using VLM
        system_prompt = "You are the Vision-Language Model.\n"
        user_prompt = f"{system_prompt}{prompt}"

        start = time.time()
        caption = generate_caption(media_path=media_path, user_prompt=user_prompt, media_type="image")
        latency_ms = int((time.time() - start) * 1000)

        # record history
        self.store.append_message(conversation_id, "user", prompt)
        self.store.append_message(conversation_id, "assistant", caption)

        return {
            "conversation_id": conversation_id,
            "media_path": media_path,
            "media_type": "image",
            "caption": caption,
            "meta": {
                "backend": "lmdeploy",
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "latency_ms": latency_ms,
            },
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
        
        answer, latency_ms = await self.convert_triplet(convo=convo, text=req.message)

        # Append to history
        self.store.append_message(conversation_id, "user", req.message)
        self.store.append_message(conversation_id, "assistant", answer)

        # Reload for updated length
        convo2 = self.store.get(conversation_id)
        
        print(f"Triplet Answer: {answer}")
        
        queries = await self.build_query(answer=json.loads(answer))
        
        retrieve = await self.RAG_faiss_retrieval(conversation_id, queries)
        
        print(f"RAG Reasoning Retrieve: {retrieve}")
        
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            # "media_attached": bool(media_path),
            "history_length": len(convo2.history) if convo2 else 0,
            "retrieve": retrieve,
            "meta": {
                "latency_ms": latency_ms,
            },
        }

    async def RAG_faiss_retrieval(self, conversation_id, text):
        """
        POST /api/v1/vlm/conversations/faiss
        JSON:
          - message: str
        """
        convo = self.store.get(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        print(f"RAG Text Input: {text}")
        image_ids, captions, scores = self.visdial_serve.faiss_search(text, self.gallery, top_k=20)
        
        anwswer = await self.reasoning(convo, text, captions)
        return {
            "id": image_ids,
            "text": captions,
            "suggest": anwswer
        }
        
    async def reasoning(self, conversation_id, input, retrieve):
        
        PROMPT_REASON = """
                Given a user query and retrieved sentences, suggest up to 3 additional (subject, relation, object) triplets that can refine the user's query.

                User query:
                {input}

                Retrieved sentences:
                {DB}

                Rules:
                - Only suggest triplets that are supported or strongly implied by the retrieved sentences.
                - Do not repeat the same information already present in the user query.
                - For each triplet, provide a short explanation of why it is useful.
                - Keep the output concise.
                Expected output:
                [{{"sug":"...","explain":"..."}}]
            """
        
        # Return format:
        #     [{{"subject":"...","relation":"...","object":"..."}}]
        prompt = PROMPT_REASON.format(
            input=input,
            DB=retrieve
        )
        
        # print(prompt)
        media_path = getattr(conversation_id, "media_path", None)
        media_type = getattr(conversation_id, "media_type", "unknown")
        history = getattr(conversation_id, "history", [])
        
        answer = generate_answer(
            media_path=media_path,
            user_prompt=prompt,
            history=history,           # role-based history; service will normalize it
            media_type=media_type,
        )
        
        suggestion = json.loads(answer)
        
        for item in suggestion:
            trip_sug = await self.convert_triplet(conversation_id, item['sug'])
            print(f"Suggestion: {item['sug']}")
            print(f"Suggestion triplet: {trip_sug[0]} Type - {type(trip_sug[0])}")
            
            item["triplet"] = trip_sug[0] #json.loads(trip_sug[0])
        
        return suggestion
    
    async def build_query(self, answer):
        
        queries = []
        for item in answer:
            subject = item.get("subject", "").strip()
            relation = item.get("relation", "").strip()
            obj = item.get("object", "").strip()
            query = f"{subject} {relation} {obj}".strip()
            queries.append(query)
        
        return "; ".join(queries)
    
    async def convert_triplet(self, convo, text):
        BASE_PROMPT = "Convert the given sentence into (subject, relation, object) triplet.\nRules:\n- Do NOT add explanations.\n- Use lowercase for relation.\nReturn format:\n[{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}]\nSentence:\n"
        # Build prompt
        PROMPT_TRIPLET = (
            "You are a Vision-Language Model.\n"
            f"User question: {BASE_PROMPT}{text}"
        )

        media_path = getattr(convo, "media_path", None)
        media_type = getattr(convo, "media_type", "unknown")
        history = getattr(convo, "history", [])
        
        print(history)

        start = time.time()
        answer = generate_answer(
            media_path=media_path,
            user_prompt=PROMPT_TRIPLET,
            history=history, # role-based history; service will normalize it
            media_type=media_type,
        )
        
        latency_ms = int((time.time() - start) * 1000)
        
        return answer, latency_ms