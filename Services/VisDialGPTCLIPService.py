import json
import logging
import os
import re
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
from Services.TargetAnnotationService import TargetAnnotationService
from Services.QVFSService import QVFS


Role = Literal["system", "user", "assistant"]
RoleHistory = List[Tuple[Role, str]]
logger = logging.getLogger("rair")


class VisDialGPTCLIPService:
    """
    VisDial retrieval service using:
    - CLIP for text embedding + FAISS retrieval
    - OpenAIService for candidate-grounded reasoning

    This class is adapted from the original VisDialCLIPService structure.
    """

    def __init__(
        self,
        vlm: str,
        device: str,
        openai_service: Optional[OpenAIService] = None,
        reasoning_model: Optional[str] = None,
        use_qvfs: bool = True,
        use_qafs: Optional[bool] = None,
        evidence_top_k: int = 3,
        fact_top_m: int = 4,
        fact_alpha: float = 0.5,
        fact_beta: float = 0.3,
        fact_gamma: float = 0.2,
        fact_delta: float = 0.5,
        rewrite_max_output_tokens: Optional[int] = None,
        reasoning_max_output_tokens: Optional[int] = None,
        compose_max_output_tokens: Optional[int] = None,
    ) -> None:
        self.vlm = vlm
        self.device = device

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.prompt = PromptCollectionService()
        self.annotation_service = TargetAnnotationService()

        # Lazily used by rewrite/diagnosis paths. Retrieval-only baselines do not
        # need an OpenAI client.
        self.openai_service = openai_service

        # Optional per-task model override
        self.reasoning_model = reasoning_model
        self.use_qvfs = use_qvfs if use_qafs is None else use_qafs
        self.evidence_top_k = evidence_top_k
        self.fact_top_m = fact_top_m
        self.fact_alpha = fact_alpha
        self.fact_beta = fact_beta
        self.fact_gamma = fact_gamma
        self.fact_delta = fact_delta
        self.rewrite_max_output_tokens = rewrite_max_output_tokens or int(
            os.environ.get("RAIR_REWRITE_MAX_TOKENS", "128")
        )
        self.reasoning_max_output_tokens = reasoning_max_output_tokens or int(
            os.environ.get("RAIR_REASONING_MAX_TOKENS", "512")
        )
        self.compose_max_output_tokens = compose_max_output_tokens or int(
            os.environ.get("RAIR_COMPOSE_MAX_TOKENS", "128")
        )

    def _get_openai_service(self) -> OpenAIService:
        if self.openai_service is None:
            self.openai_service = OpenAIService(model_name=self.reasoning_model or "gpt-5.4-mini")
        return self.openai_service

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
        if not torch.is_tensor(feats):
            if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
                feats = feats.text_embeds
            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
                feats = feats.last_hidden_state[:, 0]
            else:
                raise TypeError(f"Unsupported CLIP text feature output: {type(feats)!r}")
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

    def build_candidate_evidence(self, image_ids, captions, scores, top_k=8):
        """
        Build candidate-grounded evidence for RAIR diagnosis.
        Each candidate is enriched with dialogue-derived visual facts when the
        annotation layer has a matching image_path.
        """
        selected_ids = image_ids[:top_k]
        annotations = self.annotation_service.get_by_image_paths(selected_ids)

        evidence = []
        for rank, (image_id, caption, score) in enumerate(
            zip(image_ids[:top_k], captions[:top_k], scores[:top_k]),
            start=1,
        ):
            annotation = annotations.get(str(image_id).replace("\\", "/").lstrip("./").strip(), {})
            evidence.append(
                {
                    "rank": rank,
                    "image_id": image_id,
                    "score": score,
                    "caption": caption,
                    "visual_facts": annotation.get("visual_facts", [])[:12],
                    "positive_facts": annotation.get("positive_facts", [])[:8],
                    "negative_facts": annotation.get("negative_facts", [])[:8],
                    "uncertain_facts": annotation.get("uncertain_facts", [])[:5],
                    "enriched_caption": annotation.get("enriched_caption"),
                }
            )

        return evidence

    def select_candidate_facts(self, query: str, candidate_evidence: List[dict]) -> List[dict]:
        if not self.use_qvfs:
            return candidate_evidence

        return QVFS(
            embedding_service=self,
            top_m=self.fact_top_m,
            alpha=self.fact_alpha,
            beta=self.fact_beta,
            gamma=self.fact_gamma,
            delta=self.fact_delta,
        ).select(query=query, evidence=candidate_evidence)

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
        if not isinstance(text, str):
            return text

        cleaned = text.strip()
        cleaned = cleaned.removeprefix("\ufeff").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except Exception as direct_error:
            decoder = json.JSONDecoder()
            starts = [idx for idx in (cleaned.find("["), cleaned.find("{")) if idx >= 0]
            for start in sorted(starts):
                try:
                    value, _ = decoder.raw_decode(cleaned[start:])
                    return value
                except Exception:
                    continue

            raise ValueError(
                f"Invalid JSON returned by LLM: {text!r}"
            ) from direct_error

    @staticmethod
    def _sanitize_suggestion_text(text: str) -> str:
        cleaned = str(text or "").strip()
        replacements = [
            (r"^(specify|determine|clarify|provide|choose|ask)\s+(whether|if|the|a|an)?\s*", ""),
            (r"^(whether|if)\s+", ""),
        ]
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip("?. ")
        return cleaned

    @staticmethod
    def _suggestion_tokens(text: str) -> set:
        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with",
            "is", "are", "was", "were", "be", "this", "that", "there", "image",
            "photo", "picture",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(token) > 2 and token not in stopwords
        }

    @classmethod
    def _is_bad_suggestion(cls, sug: str, current_query: str = "") -> bool:
        text = str(sug or "").strip().lower()
        if not text:
            return True

        bad_patterns = [
            r"^(ignore|remove|exclude|avoid)\b",
            r"\bunrelated\b",
            r"\bdistractor\b",
            r"^no\s+unrelated\b",
            r"^(specify|determine|clarify|provide|choose|ask)\b",
        ]
        if any(re.search(pattern, text) for pattern in bad_patterns):
            return True

        suggestion_tokens = cls._suggestion_tokens(text)
        query_tokens = cls._suggestion_tokens(current_query)
        if suggestion_tokens and suggestion_tokens.issubset(query_tokens):
            return True

        if suggestion_tokens and query_tokens:
            overlap = len(suggestion_tokens & query_tokens) / len(suggestion_tokens)
            if overlap >= 0.85:
                return True

        return False

    @staticmethod
    def _dedupe_suggestions(suggestions: Any, current_query: str = "") -> List[dict]:
        if not isinstance(suggestions, list):
            return []

        seen = set()
        deduped = []
        for item in suggestions:
            if not isinstance(item, dict):
                continue

            sug = VisDialGPTCLIPService._sanitize_suggestion_text(
                item.get("sug")
                or item.get("suggestion")
                or item.get("query")
                or item.get("text")
                or item.get("refinement")
                or ""
            )
            normalized_key = " ".join(sorted(VisDialGPTCLIPService._suggestion_tokens(sug)))
            if (
                not sug
                or sug in seen
                or normalized_key in seen
                or VisDialGPTCLIPService._is_bad_suggestion(sug, current_query)
            ):
                continue

            seen.add(sug)
            if normalized_key:
                seen.add(normalized_key)
            deduped.append(
                {
                    "sug": sug,
                    "type": str(item.get("type", "add_detail") or "add_detail").strip(),
                    "explain": str(
                        item.get("explain")
                        or item.get("explanation")
                        or item.get("reason")
                        or ""
                    ).strip(),
                }
            )

        return deduped

    def _normalize_reasoning_output(self, data: Any, current_query: str = "") -> dict:
        """
        Normalize the RAIR reasoning response.
        The expected shape is {"diagnosis": {...}, "suggestions": [...]}, but this
        also accepts the old list-only suggestion shape as a fallback.
        """
        if isinstance(data, list):
            return {
                "diagnosis": {},
                "suggestions": self._dedupe_suggestions(data, current_query=current_query),
            }

        if not isinstance(data, dict):
            raise ValueError(f"Reasoning response must be a JSON object: {data!r}")

        diagnosis = data.get("diagnosis") or {}
        if not isinstance(diagnosis, dict):
            diagnosis = {}

        suggestions = self._dedupe_suggestions(
            data.get("suggestions", []),
            current_query=current_query,
        )
        return {
            "diagnosis": diagnosis,
            "suggestions": suggestions,
        }

    @staticmethod
    def _compact_json(data: Any) -> str:
        return json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _compact_candidate_evidence_for_prompt(candidate_evidence: Any) -> Any:
        """
        Keep only reasoning-facing fields. This avoids sending QVFS score traces,
        embeddings, or verbose metadata to the LLM prompt.
        """
        if not isinstance(candidate_evidence, list):
            return candidate_evidence

        compact = []
        for candidate in candidate_evidence:
            if not isinstance(candidate, dict):
                continue
            compact.append(
                {
                    "rank": candidate.get("rank"),
                    "caption": candidate.get("caption"),
                    "visual_facts": (candidate.get("visual_facts") or [])[:6],
                    "positive_facts": (candidate.get("positive_facts") or [])[:4],
                    "negative_facts": (candidate.get("negative_facts") or [])[:3],
                    "uncertain_facts": (candidate.get("uncertain_facts") or [])[:2],
                }
            )
        return compact

    async def rewrite_query(self, context_state: Any):
        """
        Rewrite the interaction context into a natural-language retrieval query.
        """
        context_json = self._compact_json(context_state)
        prompt = self.prompt.rewrite_context.format(context=context_json)

        start = time.time()
        answer = self._get_openai_service().generate_answer(
            user_prompt=prompt,
            history=None,
            model=self.reasoning_model,
            max_output_tokens=self.rewrite_max_output_tokens,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        data = self._safe_json_loads(answer)
        rewritten_query = str(data.get("rewritten_query", "")).strip()
        if not rewritten_query:
            raise ValueError(f"Missing rewritten_query in LLM response: {answer}")

        return rewritten_query, latency_ms

    async def compose_refined_query(
        self,
        current_query: str,
        accepted_suggestion: Any,
    ) -> Tuple[str, int]:
        """
        Compose a concise retrieval query from the current query and an accepted
        RAIR suggestion. This is safer for CLIP than raw string appending.
        """
        if not accepted_suggestion:
            return current_query, 0

        if isinstance(accepted_suggestion, dict):
            suggestion_text = str(accepted_suggestion.get("sug", "")).strip()
        else:
            suggestion_text = str(accepted_suggestion or "").strip()

        if not suggestion_text:
            return current_query, 0

        prompt = self.prompt.compose_refinement.format(
            current_query=current_query,
            accepted_suggestion=suggestion_text,
        )

        start = time.time()
        answer = self._get_openai_service().generate_answer(
            user_prompt=prompt,
            history=None,
            model=self.reasoning_model,
            max_output_tokens=self.compose_max_output_tokens,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        data = self._safe_json_loads(answer)
        refined_query = str(data.get("refined_query", "")).strip()
        if not refined_query:
            logger.warning("Missing refined_query in composer response: %s", answer)
            return f"{current_query}; {suggestion_text}", latency_ms

        return refined_query, latency_ms

    async def RAG_faiss_retrieval(self, history, gallery, text):
        """
        1. Retrieve by FAISS
        2. Ask OpenAI to generate candidate-grounded query refinement suggestions
        """
        image_ids, captions, scores = self.faiss_search(text, gallery, top_k=20)
        candidate_evidence_raw = self.build_candidate_evidence(
            image_ids=image_ids,
            captions=captions,
            scores=scores,
            top_k=self.evidence_top_k,
        )
        candidate_evidence = self.select_candidate_facts(
            query=text,
            candidate_evidence=candidate_evidence_raw,
        )
        logger.info(
            "RAIR candidate evidence top5 use_qvfs=%s:\n%s",
            self.use_qvfs,
            json.dumps(candidate_evidence[:5], ensure_ascii=False, indent=2),
        )

        reasoning_result = await self.reasoning(history, text, candidate_evidence)
        logger.info(
            "RAIR diagnosis:\n%s",
            json.dumps(reasoning_result.get("diagnosis", {}), ensure_ascii=False, indent=2),
        )
        logger.info(
            "RAIR suggestions:\n%s",
            json.dumps(reasoning_result.get("suggestions", []), ensure_ascii=False, indent=2),
        )

        return {
            "id": image_ids,
            "text": captions,
            "score": scores,
            "candidate_evidence": candidate_evidence,
            "candidate_evidence_raw": candidate_evidence_raw,
            "fact_selection": {
                "method": "qvfs" if self.use_qvfs else "none",
                "evidence_top_k": self.evidence_top_k,
                "top_m": self.fact_top_m,
                "alpha": self.fact_alpha,
                "beta": self.fact_beta,
                "gamma": self.fact_gamma,
                "delta": self.fact_delta,
            },
            "diagnosis": reasoning_result.get("diagnosis", {}),
            "suggest": reasoning_result.get("suggestions", []),
        }

    async def reasoning(self, history, input, candidate_evidence):
        """
        Use OpenAI to diagnose retrieved candidates and generate refinements.
        """
        prompt_evidence = self._compact_candidate_evidence_for_prompt(candidate_evidence)
        prompt = self.prompt.reason.format(
            input_query=input,
            db=self._compact_json(prompt_evidence),
        )

        answer = self._get_openai_service().generate_answer(
            user_prompt=prompt,
            history=history,
            model=self.reasoning_model,
            max_output_tokens=self.reasoning_max_output_tokens,
            # temperature=0.2,
            store=False,
        )

        data = self._safe_json_loads(answer)
        normalized = self._normalize_reasoning_output(data, current_query=input)
        if not normalized.get("suggestions"):
            logger.warning("RAIR LLM returned no usable suggestions. Raw response: %s", answer)
            logger.warning("RAIR prompt evidence: %s", self._compact_json(prompt_evidence))
        return normalized
