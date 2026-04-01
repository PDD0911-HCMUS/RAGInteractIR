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

from Services.QwenVLService import generate_answer
# from Services.VisDialSigLIPService import VisDialSiGLIPService
from Services.VisDialCLIPService import VisDialCLIPService


# ---- typing shared ----
Role = Literal["user", "assistant", "system"]

class MessageRequest(BaseModel):
    message: str

# ---- Storage interface (so you can swap implementations easily) ----
class ConversationProtocol(Protocol):
    conversation_id: str
    history: list[tuple[Role, str]]

class ConversationStoreProtocol(Protocol):
    def create(self, conversation_id: str) -> ConversationProtocol: ...
    def get(self, conversation_id: str) -> Optional[ConversationProtocol]: ...
    def append_message(self, conversation_id: str, role: Role, content: str) -> ConversationProtocol: ...


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
        caption = generate_answer(user_prompt=greeting_prompt, history=None)
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
        
        answer, latency_ms = await self.convert_triplet(text=req.message)

        # Append to history
        self.store.append_message(conversation_id, "user", req.message)
        self.store.append_message(conversation_id, "assistant", answer)

        # Reload for updated length
        convo2 = self.store.get(conversation_id)
        
        print(f"Triplet Answer: {answer}")
        
        queries = await self.build_query(answer=json.loads(answer))
        
        retrieve = await self.RAG_faiss_retrieval(conversation_id, queries)
        
        # print(f"RAG Reasoning Retrieve: {retrieve}")
        
        return {
            "conversation_id": conversation_id,
            "answer": answer,
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
        image_ids, captions, scores = self.visdial_serve.faiss_search(text, self.gallery, top_k=100)
        
        anwswer = await self.reasoning(convo, text, captions)
        return {
            "id": image_ids,
            "text": captions,
            "suggest": anwswer
        }
        
    async def reasoning(self, conversation_id, input, retrieve):
        
        PROMPT_REASON = """
You are a retrieval refinement assistant.

Your task is to improve the user's image search query by proposing exactly 3 short query suggestions based on the retrieved captions.

The goal is NOT to answer the user.
The goal is NOT to rewrite the query with synonyms.
The goal is to suggest 3 better follow-up queries that help the search system retrieve images closer to the user's true intention.

User query:
{input}

Retrieved captions:
{DB}

Rules:
1. Use only details explicitly supported or strongly implied by the retrieved captions.
2. Keep the user's original intent unchanged.
3. Each suggestion must add NEW visual detail not already present in the user query.
4. The 3 suggestions must be meaningfully different from each other.
5. Each suggestion must focus on a different refinement aspect whenever possible, such as:
   - action
   - scene/background
   - spatial relation
   - nearby object
   - attribute
   - count
6. Do NOT produce multiple suggestions with the same core meaning.
7. If two suggestions describe the same main fact, keep only the more specific one.
8. Prefer details that are repeated or consistent across multiple retrieved captions.
9. If the captions are noisy or contradictory, use only the safest supported details.
10. Do NOT hallucinate.
11. Do NOT answer the user.
12. Do NOT summarize the captions.
13. Do NOT repeat or paraphrase the user query.
14. Keep each suggestion short and directly usable as a search query.
15. Each explanation must clearly state what NEW detail was added.

Output requirements:
- Return valid JSON only.
- Return exactly 3 items.
- Do not include markdown fences.
- Do not include any text before or after the JSON.

Output format:
[
  {{"sug":"...", "explain":"..."}},
  {{"sug":"...", "explain":"..."}},
  {{"sug":"...", "explain":"..."}}
]
"""
        
        # Return format:
        #     [{{"subject":"...","relation":"...","object":"..."}}]
        prompt = PROMPT_REASON.format(
            input=input,
            DB=retrieve
        )
        
        # print(prompt)
        history = getattr(conversation_id, "history", [])
        
        answer = generate_answer(
            user_prompt=prompt,
            history=history
        )
        
        print(f"RAW Answer Reasoning: {answer}")
        suggestion = json.loads(answer)
        
        seen = set()
        suggestion = [d for d in suggestion if not (d["sug"] in seen or seen.add(d["sug"]))]
        
        print(f"RAW Answer Reasoning Remove Dup: {answer}")
        
        for item in suggestion:
            trip_sug = await self.convert_triplet(item['sug'])
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
    
    async def convert_triplet(self, text):
        print(f"Input Convert Triplet: {text}")
        BASE_PROMPT = "Convert the given sentence into (subject, relation, object) triplet.\nRules:\n- Do NOT add explanations.\n- Use lowercase for relation.\nReturn format:\n[{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}]\nSentence:\n"
        # Build prompt
        PROMPT_TRIPLET = (
            "You are a Vision-Language Model.\n"
            f"User question: {BASE_PROMPT}{text}"
        )

        start = time.time()
        answer = generate_answer(
            user_prompt=PROMPT_TRIPLET,
            history=None, # role-based history; service will normalize it
        )
        
        latency_ms = int((time.time() - start) * 1000)
        
        return answer, latency_ms