import json
import logging
import os
import time
from typing import Optional, List, Tuple, Literal, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
from sqlalchemy import select

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from Database.db_session import SessionLocal
from Entities.entities import VisDialCLIPCapDial
from Services.PromptCollectionService import PromptCollectionService
from Services.OpenAIService import OpenAIService


Role = Literal["system", "user", "assistant"]
RoleHistory = List[Tuple[Role, str]]
logger = logging.getLogger("rair")


class VisDialGPTCLIPService:
    """
    VisDial retrieval service using:
    - CLIP for text embedding + FAISS retrieval
    - OpenAIService for triplet conversion and reasoning

    This class is adapted from the original VisDialCLIPService structure.
    """

    def __init__(
        self,
        vlm: str,
        device: str,
        openai_service: Optional[OpenAIService] = None,
        triplet_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
    ) -> None:
        self.vlm = vlm
        self.device = device

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.prompt = PromptCollectionService()

        # Shared OpenAI service instance
        self.openai_service = openai_service or OpenAIService(model_name="gpt-5.4-mini")

        # Optional per-task model override
        self.triplet_model = triplet_model
        self.reasoning_model = reasoning_model

    def create_vlm(self):
        local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0"
        tokenizer = AutoTokenizer.from_pretrained(
            self.vlm,
            local_files_only=local_files_only,
        )
        model = CLIPModel.from_pretrained(
            self.vlm,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        processor = AutoProcessor.from_pretrained(
            self.vlm,
            local_files_only=local_files_only,
        )
        return tokenizer, processor, model

    def _ensure_vlm_loaded(self):
        if self.tokenizer is None or self.processor is None or self.model is None:
            self.tokenizer, self.processor, self.model = self.create_vlm()

    def preload_model(self):
        self._ensure_vlm_loaded()

    @torch.no_grad()
    def embed_texts(self, texts):
        """
        Encode text list into normalized CLIP text embeddings.
        """
        self._ensure_vlm_loaded()
        inputs = self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        feats = self.model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)

        return feats.cpu()

    def build_gallery(self):
        """
        Build FAISS gallery from VisDialCLIPCapDial table.
        """
        import faiss

        console = Console()
        start_total = time.perf_counter()

        console.rule("[bold cyan]Creating VisDial Gallery + FAISS[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Initializing...", total=5)

            # Step 1: Load rows from DB
            progress.update(task, description="Loading rows from database...", completed=0)
            t0 = time.perf_counter()
            with SessionLocal() as session:
                rows = session.execute(
                    select(
                        VisDialCLIPCapDial.image_path,
                        VisDialCLIPCapDial.caption,
                        VisDialCLIPCapDial.img_em,
                        VisDialCLIPCapDial.cap_em,
                    )
                ).all()
            t1 = time.perf_counter()
            progress.advance(task)

            if not rows:
                console.print("[bold red]No data found in VisDialCLIPCapDial.[/bold red]")
                raise ValueError("No data found in VisDialCLIPCapDial.")

            # Step 2: Extract metadata
            progress.update(task, description="Extracting image paths and captions...", completed=1)
            t2 = time.perf_counter()
            image_ids = [row.image_path for row in rows]
            captions = [row.caption for row in rows]
            t3 = time.perf_counter()
            progress.advance(task)

            # Step 3: Convert embeddings to numpy
            progress.update(task, description="Converting embeddings to numpy arrays...", completed=2)
            t4 = time.perf_counter()
            img_embeddings = np.array([row.img_em for row in rows], dtype=np.float32)
            cap_embeddings = np.array([row.cap_em for row in rows], dtype=np.float32)

            if cap_embeddings.ndim == 3 and cap_embeddings.shape[1] == 1:
                cap_embeddings = np.squeeze(cap_embeddings, axis=1)

            t5 = time.perf_counter()
            progress.advance(task)

            # Step 4: Build FAISS indices
            progress.update(task, description="Preparing FAISS indices...", completed=3)
            t6 = time.perf_counter()

            # Uncomment if your embeddings are NOT already normalized
            # faiss.normalize_L2(img_embeddings)
            # faiss.normalize_L2(cap_embeddings)

            dim_img = img_embeddings.shape[1]
            dim_cap = cap_embeddings.shape[1]

            img_index = faiss.IndexFlatIP(dim_img)
            cap_index = faiss.IndexFlatIP(dim_cap)

            img_index.add(img_embeddings)
            cap_index.add(cap_embeddings)

            t7 = time.perf_counter()
            progress.advance(task)

            # Step 5: Finalize gallery
            progress.update(task, description="Finalizing gallery object...", completed=4)
            t8 = time.perf_counter()
            gallery = {
                "image_id": image_ids,
                "caption": captions,
                "img_em": img_embeddings,
                "cap_em": cap_embeddings,
                "img_index": img_index,
                "cap_index": cap_index,
            }
            t9 = time.perf_counter()
            progress.advance(task)

        summary = Table(title="Gallery Build Summary", show_lines=False)
        summary.add_column("Field", style="cyan", no_wrap=True)
        summary.add_column("Value", style="green")

        summary.add_row("Loaded rows", f"{len(rows):,}")
        summary.add_row("Image embedding shape", str(img_embeddings.shape))
        summary.add_row("Caption embedding shape", str(cap_embeddings.shape))
        summary.add_row("FAISS image index total", f"{img_index.ntotal:,}")
        summary.add_row("FAISS caption index total", f"{cap_index.ntotal:,}")
        summary.add_row("DB load time", f"{t1 - t0:.2f}s")
        summary.add_row("Metadata extract time", f"{t3 - t2:.2f}s")
        summary.add_row("Numpy conversion time", f"{t5 - t4:.2f}s")
        summary.add_row("FAISS build time", f"{t7 - t6:.2f}s")
        summary.add_row("Finalize time", f"{t9 - t8:.2f}s")
        summary.add_row("Total time", f"{time.perf_counter() - start_total:.2f}s")

        console.print(summary)
        console.rule("[bold green]Build done[/bold green]")

        return gallery

    def faiss_search(self, query_text, gallery, top_k=20):
        """
        Search image index using query text embedding.
        """
        q = self.embed_texts(query_text)
        q = np.array(q, dtype=np.float32).reshape(1, -1)

        # If needed:
        # faiss.normalize_L2(q)

        scores, indices = gallery["img_index"].search(q, top_k)

        image_ids, captions, s = [], [], []
        for score, idx in zip(scores[0], indices[0]):
            image_ids.append(gallery["image_id"][idx])
            captions.append(gallery["caption"][idx])
            s.append(float(score))

        return image_ids, captions, s

    async def build_query(self, answer):
        """
        Convert list[dict] triplets into a compact retrieval query string.
        Example:
        [{"subject":"dog","relation":"on","object":"chair"}]
        -> "dog on chair"
        """
        queries = []
        for item in answer:
            subject = item.get("subject", "").strip()
            relation = item.get("relation", "").strip()
            obj = item.get("object", "").strip()

            query = f"{subject} {relation} {obj}".strip()
            if query:
                queries.append(query)

        return "; ".join(queries)

    def build_dial(self, suggestion, query):
        """
        - Ta cần xác định được đâu là respond của LLMs và query User.
        - Ta sẽ nối lại thành (Sg_i, Query_i, ...) -> được gọi là 1 context dialouge.
        - Việc cần xử lý ở FE sẽ là thu thập các cấu trúc đầu vào.
        - Sử dụng một Promt LLMs để xác đinh:
            - sự thay đổi của hội thoại
            - trích xuất keywords
            - tóm tắt nội dung
            -> CoT để re-write lại thành 1 query
            -> sử dụng cái này để suy luận
        """
        
        
        pass
    
    def _safe_json_loads(self, text: str) -> Any:
        """
        Safe JSON parsing helper with a clearer error.
        """
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid JSON returned by LLM: {text}") from e

    async def rewrite_query(self, context_state: Any):
        """
        Rewrite the interaction context into a natural-language retrieval query.
        """
        context_json = json.dumps(context_state, ensure_ascii=False, indent=2)
        prompt = self.prompt.rewrite_context.format(context=context_json)

        start = time.time()
        answer = self.openai_service.generate_answer(
            user_prompt=prompt,
            history=None,
            model=self.reasoning_model,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        data = self._safe_json_loads(answer)
        rewritten_query = str(data.get("rewritten_query", "")).strip()
        if not rewritten_query:
            raise ValueError(f"Missing rewritten_query in LLM response: {answer}")

        return rewritten_query, latency_ms

    async def convert_triplet(self, text: str, history: Optional[RoleHistory]):
        """
        Use OpenAI to convert user text into triplet JSON.
        """

        prompt = self.prompt.convert_triplet.format(text=text)

        start = time.time()
        answer = self.openai_service.generate_answer(
            user_prompt=prompt,
            history=history,
            model=self.triplet_model,
            # Optional: keep the model focused and cheap
            # temperature=0.0,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        return answer, latency_ms

    async def RAG_faiss_retrieval(self, history, gallery, text, triplets=None):
        """
        1. Retrieve by FAISS
        2. Ask OpenAI to generate query refinement suggestions
        """
        image_ids, captions, scores = self.faiss_search(text, gallery, top_k=20)
        top_results = [
            {
                "rank": rank,
                "image_id": image_id,
                "score": score,
                "caption": caption,
            }
            for rank, (image_id, score, caption) in enumerate(
                zip(image_ids[:5], scores[:5], captions[:5]),
                start=1,
            )
        ]
        logger.info(
            "RAIR retrieve top5:\n%s",
            json.dumps(top_results, ensure_ascii=False, indent=2),
        )

        answer = await self.reasoning(history, text, captions, triplets=triplets)
        logger.info(
            "RAIR suggestions:\n%s",
            json.dumps(answer, ensure_ascii=False, indent=2),
        )

        return {
            "id": image_ids,
            "text": captions,
            "score": scores,
            "suggest": answer
        }

    async def reasoning(self, history, input, retrieve, triplets=None):
        """
        Use OpenAI to generate refinement suggestions based on retrieved captions.
        """
        prompt = self.prompt.reason.format(
            input_query=input,
            triplets=json.dumps(triplets or [], ensure_ascii=False),
            db=retrieve
        )

        answer = self.openai_service.generate_answer(
            user_prompt=prompt,
            history=history,
            model=self.reasoning_model,
            # temperature=0.2,
            store=False,
        )

        suggestion = self._safe_json_loads(answer)

        # Remove duplicated "sug"
        seen = set()
        suggestion = [
            d for d in suggestion
            if not (d.get("sug") in seen or seen.add(d.get("sug")))
        ]

        return suggestion
