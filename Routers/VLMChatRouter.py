import os
import uuid
import time
from typing import Optional, Protocol, Literal
import json
from fastapi import APIRouter, File, Form, HTTPException
from pydantic import BaseModel

from Services.QwenVLService import generate_answer
from Services.PromptCollectionService import PromptCollectionService
from Services.VisDialCLIPService import VisDialCLIPService
from Storage.ConversationStore import ConversationStore


# ---- typing shared ----
Role = Literal["user", "assistant", "system"]

class MessageRequest(BaseModel):
    message: str

class VLMChatRouter:
    def __init__(
        self,
        conversation_store: ConversationStore,
        upload_dir: Optional[str] = None,
    ):
        #--------------Config---------------#
        self.store = conversation_store
        self.upload_dir = upload_dir or os.environ.get("VLM_UPLOAD_DIR", "./uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self.visdial_serve = VisDialCLIPService(
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device="cpu",
            vlm="openai/clip-vit-base-patch32"
        )
        self.prompt = PromptCollectionService()
        self.router = APIRouter(prefix="/vlm", tags=["VLM"])
        self.gallery = self.visdial_serve.build_gallery()
        
        #----------------API----------------#

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

    # ----------------- ENPOINTS -----------------
    
    """
    Theo luồng là:
    1. input user
    2. Convert triplet (PROMPT CONVERT)
    3. Embedding
    4. RAG-Reasoning:
        4.1. FAISS
        4.2. Collect top-k caption & images
        4.3. cho LLM suy luận (PROMPT REASON)
        4.4. trả về tầm 3 suggestions 
    5. Trả về hình ảnh và suggestions
    
    LƯU Ý: nhớ để ý đến HISTORY để lưu được quá trình truy vấn để khớp với user intention.
    """
    
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
        caption = generate_answer(user_prompt=self.prompt.greeting, history=None)
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
        
        print(f"User Message: {req.message}")
        
        history = convo.history
        # CONVERT TRIPLETS (Step 2.0)
        triplets, latency_ms = await self.convert_triplet(text=req.message, history=history)
        
        print(f"History Chat: {history}")
        print(f"Triplet Answer: {triplets}")
        
        queries = await self.build_query(answer=json.loads(triplets))
        print(f"Queries Built: {queries}")
        
        # Append to history
        self.store.append_message(conversation_id, "user", req.message)
        # self.store.append_message(conversation_id, "assistant", triplets)
        
        # Update convo
        convo2 = self.store.get(conversation_id)
        
        retrieve = await self.RAG_faiss_retrieval(convo2.history, queries)
        
        assistant_payload = json.dumps({
            "triplets": triplets,
            "suggestions": retrieve.get("suggest", [])
        }, ensure_ascii=False)
        
        self.store.append_message(conversation_id, "assistant", assistant_payload)
        
        return {
            "conversation_id": conversation_id,
            "answer": triplets,
            "history_length": len(history) if convo else 0,
            "retrieve": retrieve,
            "meta": {
                "latency_ms": latency_ms,
            },
        }

    async def RAG_faiss_retrieval(self, history, text):
        """
        POST /api/v1/vlm/conversations/faiss
        JSON:
          - message: str
        Nội dung:
          - Hàm này sẽ gọi 2 phương thức:
            + faiss_search: truy vấn kết quả từ DB với truy vấn từ user
            + reasoning: sử dụng kết quả truy vấn để suy luận và đưa ra suggestions
        """

        print(f"RAG Text Input: {text}")
        image_ids, captions, scores = self.visdial_serve.faiss_search(text, self.gallery, top_k=20)
        
        anwswer = await self.reasoning(history, text, captions)
        
        return {
            "id": image_ids,
            "text": captions,
            "suggest": anwswer
        }
        
    async def reasoning(self, history, input, retrieve):
        
        prompt = self.prompt.reason.format(
            input_query=input,
            db=retrieve
        )
        
        # print(prompt)
        print(f"Reasoning History: {history}")
        
        answer = generate_answer(
            user_prompt=prompt,
            history=history
        )
        
        # print(f"RAW Answer Reasoning: {answer}")
        suggestion = json.loads(answer)
        
        seen = set()
        suggestion = [d for d in suggestion if not (d["sug"] in seen or seen.add(d["sug"]))]
        
        # print(f"RAW Answer Reasoning Remove Dup: {answer}")
        
        for item in suggestion:
            trip_sug = await self.convert_triplet(item['sug'], history)
            
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
    
    async def convert_triplet(self, text, history):
        print(f"Input Convert Triplet: {text}")
        
        prompt = self.prompt.convert_triplet.format(
            text=text
        )

        start = time.time()
        answer = generate_answer(
            user_prompt=prompt,
            history=history, # role-based history; service will normalize it
        )
        
        latency_ms = int((time.time() - start) * 1000)
        
        return answer, latency_ms