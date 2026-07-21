import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Services.VisDialGPTCLIPService import VisDialGPTCLIPService


logger = logging.getLogger("rair.api")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int env %s=%r; using %d", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using %.3f", name, value, default)
        return default


def _resolve_device(env_name: str, default: str = "cpu") -> str:
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


@dataclass
class RAIRSession:
    session_id: str
    embedding_backend: str
    embedding_model: str
    retrieval_index: str
    fusion_alpha: float
    fusion_pool_size: int
    result_top_k: int
    initial_query: str = ""
    current_query: str = ""
    turn: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    feedback_pairs: List[Dict[str, Any]] = field(default_factory=list)
    pending_suggestions: List[Dict[str, Any]] = field(default_factory=list)


class RAIRInteractiveRetrievalService:
    """
    Stateful RAIR-VF pipeline for API usage.

    This is the online counterpart of E3: user query -> rewrite -> retrieval
    with image/caption/fusion FAISS -> QVFS -> candidate-grounded diagnosis and
    suggestions. It keeps only lightweight in-memory session state.
    """

    def __init__(self) -> None:
        self.embedding_backend = os.environ.get("RAIR_EMBEDDING_BACKEND", "siglip")
        self.embedding_mode = os.environ.get("RAIR_EMBEDDING_MODE", "train")
        self.embedding_model = os.environ.get(
            "RAIR_EMBEDDING_MODEL",
            "google/siglip-base-patch16-224"
            if self.embedding_backend.lower() == "siglip"
            else "openai/clip-vit-base-patch32",
        )
        self.device = _resolve_device("RAIR_EMBEDDING_DEVICE", _resolve_device("CLIP_DEVICE"))

        self.llm_provider = os.environ.get("RAIR_LLM_PROVIDER", "local").lower()
        self.reasoning_model = os.environ.get("RAIR_REASONING_MODEL", "google/gemma-3-12b-it")
        self.local_llm_device = os.environ.get("LOCAL_LLM_DEVICE", "cuda")
        self.local_llm_dtype = os.environ.get("LOCAL_LLM_DTYPE", "bfloat16")

        self.use_qvfs = _env_bool("RAIR_USE_QVFS", _env_bool("RAIR_USE_QAFS", True))
        self.evidence_top_k = _env_int("RAIR_EVIDENCE_TOP_K", 10)
        self.fact_top_m = _env_int("RAIR_FACT_TOP_M", 4)
        self.fact_alpha = _env_float("RAIR_FACT_ALPHA", 0.5)
        self.fact_beta = _env_float("RAIR_FACT_BETA", 0.3)
        self.fact_gamma = _env_float("RAIR_FACT_GAMMA", 0.2)
        self.fact_delta = _env_float("RAIR_FACT_DELTA", 0.5)

        self.retrieval_index = os.environ.get("RAIR_RETRIEVAL_INDEX", "fusion")
        self.fusion_alpha = _env_float("RAIR_FUSION_ALPHA", 0.9)
        self.fusion_pool_size = _env_int("RAIR_FUSION_POOL_SIZE", 200)
        self.search_depth = _env_int("RAIR_SEARCH_DEPTH", 50)
        self.result_top_k = _env_int("RAIR_RESULT_TOP_K", 20)

        self.rewrite_max_tokens = _env_int("RAIR_REWRITE_MAX_TOKENS", 128)
        self.reasoning_max_tokens = _env_int("RAIR_REASONING_MAX_TOKENS", 512)
        self.compose_max_tokens = _env_int("RAIR_COMPOSE_MAX_TOKENS", 128)

        self._llm_service: Optional[Any] = None
        self._retrieval_services: Dict[str, VisDialGPTCLIPService] = {}
        self._galleries: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, RAIRSession] = {}

    @staticmethod
    def _normalize_embedding_backend(value: Optional[str]) -> str:
        backend = str(value or "siglip").strip().lower()
        aliases = {
            "clip": "clip",
            "openai_clip": "clip",
            "siglip": "siglip",
            "sig-lip": "siglip",
            "google_siglip": "siglip",
        }
        if backend not in aliases:
            raise ValueError(f"Unsupported embedding_backend: {value}")
        return aliases[backend]

    @staticmethod
    def _normalize_retrieval_index(value: Optional[str]) -> str:
        index = str(value or "fusion").strip().lower()
        aliases = {
            "image": "image",
            "img": "image",
            "images": "image",
            "caption": "caption",
            "cap": "caption",
            "captions": "caption",
            "fusion": "fusion",
            "both": "fusion",
            "hybrid": "fusion",
        }
        if index not in aliases:
            raise ValueError(f"Unsupported retrieval_index: {value}")
        return aliases[index]

    def _default_model_for_backend(self, embedding_backend: str) -> str:
        if embedding_backend == "siglip":
            return "google/siglip-base-patch16-224"
        return "openai/clip-vit-base-patch32"

    def _resolve_embedding_model(
        self,
        embedding_backend: str,
        embedding_model: Optional[str] = None,
    ) -> str:
        if embedding_model:
            return str(embedding_model)

        default_backend = self._normalize_embedding_backend(self.embedding_backend)
        if embedding_backend == default_backend and self.embedding_model:
            return str(self.embedding_model)

        return self._default_model_for_backend(embedding_backend)

    def _backend_key(self, embedding_backend: str, embedding_model: str) -> str:
        return f"{embedding_backend}|{embedding_model}|{self.embedding_mode}"

    def _get_llm_service(self) -> Any:
        if self._llm_service is not None:
            return self._llm_service

        if self.llm_provider == "openai":
            from Services.OpenAIService import OpenAIService

            self._llm_service = OpenAIService(model_name=self.reasoning_model)
        else:
            from Services.LocalLLMService import LocalLLMService

            self._llm_service = LocalLLMService(
                model_name=self.reasoning_model,
                device=self.local_llm_device,
                dtype=self.local_llm_dtype,
                max_new_tokens=max(
                    self.rewrite_max_tokens,
                    self.reasoning_max_tokens,
                    self.compose_max_tokens,
                ),
            )
        return self._llm_service

    def _get_retrieval_service(
        self,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> VisDialGPTCLIPService:
        backend = self._normalize_embedding_backend(embedding_backend or self.embedding_backend)
        model_name = self._resolve_embedding_model(backend, embedding_model)
        key = self._backend_key(backend, model_name)

        if key not in self._retrieval_services:
            self._retrieval_services[key] = VisDialGPTCLIPService(
                vlm=model_name,
                device=self.device,
                openai_service=self._get_llm_service(),
                reasoning_model=self.reasoning_model,
                use_qvfs=self.use_qvfs,
                evidence_top_k=self.evidence_top_k,
                fact_top_m=self.fact_top_m,
                fact_alpha=self.fact_alpha,
                fact_beta=self.fact_beta,
                fact_gamma=self.fact_gamma,
                fact_delta=self.fact_delta,
                rewrite_max_output_tokens=self.rewrite_max_tokens,
                reasoning_max_output_tokens=self.reasoning_max_tokens,
                compose_max_output_tokens=self.compose_max_tokens,
                embedding_backend=backend,
                embedding_mode=self.embedding_mode,
            )
        return self._retrieval_services[key]

    def get_gallery(
        self,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        backend = self._normalize_embedding_backend(embedding_backend or self.embedding_backend)
        model_name = self._resolve_embedding_model(backend, embedding_model)
        key = self._backend_key(backend, model_name)

        if key not in self._galleries:
            logger.info(
                "Building RAIR gallery backend=%s model=%s mode=%s",
                backend,
                model_name,
                self.embedding_mode,
            )
            self._galleries[key] = self._get_retrieval_service(
                embedding_backend=backend,
                embedding_model=model_name,
            ).build_gallery()
        return self._galleries[key]

    def preload(self) -> None:
        if _env_bool("RAIR_PRELOAD_MODEL", False):
            self._get_retrieval_service().preload_model()
        if _env_bool("RAIR_PRELOAD_GALLERY", True):
            self.get_gallery()
        if _env_bool("RAIR_PRELOAD_LLM", True):
            self.preload_llm()

    def preload_llm(self) -> None:
        """Load the local reasoning LLM during API startup when possible."""
        start = time.perf_counter()
        llm_service = self._get_llm_service()
        if hasattr(llm_service, "_load"):
            logger.info(
                "Preloading RAIR local LLM provider=%s model=%s device=%s dtype=%s",
                self.llm_provider,
                self.reasoning_model,
                self.local_llm_device,
                self.local_llm_dtype,
            )
            llm_service._load()
        else:
            logger.info(
                "RAIR LLM provider=%s uses lazy remote/client initialization; skipping model warm-up",
                self.llm_provider,
            )
        logger.info("RAIR LLM preload done elapsed=%.2fs", time.perf_counter() - start)

    @staticmethod
    def _compact_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for item in suggestions or []:
            if not isinstance(item, dict):
                continue
            sug = str(item.get("sug", "")).strip()
            if not sug:
                continue
            entry = {"sug": sug}
            for key in ("type", "explain"):
                value = str(item.get(key, "") or "").strip()
                if value:
                    entry[key] = value
            compact.append(entry)
        return compact

    @staticmethod
    def _media_url(image_id: Any) -> str:
        path = str(image_id or "").replace("\\", "/").lstrip("./")
        return f"/media/{path}"

    def _context_state(
        self,
        session: RAIRSession,
        latest_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "initial_query": session.initial_query,
            "feedback_pairs": session.feedback_pairs,
            "pending_suggestions": session.pending_suggestions,
            "latest_user_message": latest_user_message,
        }

    def _format_results(
        self,
        image_ids: List[Any],
        captions: List[str],
        scores: List[float],
        result_top_k: int,
    ) -> List[Dict[str, Any]]:
        results = []
        for rank, (image_id, caption, score) in enumerate(
            zip(image_ids[:result_top_k], captions[:result_top_k], scores[:result_top_k]),
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "image_id": image_id,
                    "caption": caption,
                    "score": float(score),
                    "media_url": self._media_url(image_id),
                }
            )
        return results

    async def _run_rair_turn(
        self,
        session: RAIRSession,
        user_message: str,
    ) -> Dict[str, Any]:
        retrieval_service = self._get_retrieval_service(
            embedding_backend=session.embedding_backend,
            embedding_model=session.embedding_model,
        )
        gallery = self.get_gallery(
            embedding_backend=session.embedding_backend,
            embedding_model=session.embedding_model,
        )
        context_state = self._context_state(session, latest_user_message=user_message)

        start = time.time()
        rewritten_query, rewrite_latency_ms = await retrieval_service.rewrite_query(context_state)
        image_ids, captions, scores = retrieval_service.faiss_search(
            query_text=rewritten_query,
            gallery=gallery,
            top_k=max(self.search_depth, session.result_top_k, self.evidence_top_k),
            retrieval_index=session.retrieval_index,
            fusion_alpha=session.fusion_alpha,
            fusion_pool_size=session.fusion_pool_size,
        )

        candidate_evidence_raw = retrieval_service.build_candidate_evidence(
            image_ids=image_ids,
            captions=captions,
            scores=scores,
            top_k=self.evidence_top_k,
        )
        candidate_evidence = retrieval_service.select_candidate_facts(
            query=rewritten_query,
            candidate_evidence=candidate_evidence_raw,
        )
        reasoning = await retrieval_service.reasoning(
            history=None,
            input=rewritten_query,
            candidate_evidence=candidate_evidence,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        suggestions = self._compact_suggestions(reasoning.get("suggestions", []))
        session.current_query = rewritten_query
        session.pending_suggestions = suggestions
        session.turn += 1

        turn_payload = {
            "turn": session.turn,
            "user_message": user_message,
            "context_state": context_state,
            "rewritten_query": rewritten_query,
            "retrieval": {
                "backend": session.embedding_backend,
                "model": session.embedding_model,
                "index": session.retrieval_index,
                "fusion_alpha": session.fusion_alpha if session.retrieval_index == "fusion" else None,
                "fusion_pool_size": session.fusion_pool_size if session.retrieval_index == "fusion" else None,
                "search_depth": self.search_depth,
                "results": self._format_results(image_ids, captions, scores, session.result_top_k),
            },
            "candidate_evidence": candidate_evidence,
            "fact_selection": {
                "method": "qvfs" if self.use_qvfs else "none",
                "evidence_top_k": self.evidence_top_k,
                "top_m": self.fact_top_m,
                "alpha": self.fact_alpha,
                "beta": self.fact_beta,
                "gamma": self.fact_gamma,
                "delta": self.fact_delta,
            },
            "diagnosis": reasoning.get("diagnosis", {}),
            "suggestions": suggestions,
            "meta": {
                "llm_provider": self.llm_provider,
                "reasoning_model": self.reasoning_model,
                "rewrite_latency_ms": rewrite_latency_ms,
                "elapsed_ms": elapsed_ms,
            },
        }
        session.history.append(turn_payload)
        return turn_payload

    def create_session(
        self,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        retrieval_index: Optional[str] = None,
        fusion_alpha: Optional[float] = None,
        fusion_pool_size: Optional[int] = None,
        result_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        backend = self._normalize_embedding_backend(embedding_backend or self.embedding_backend)
        model_name = str(embedding_model or self._default_model_for_backend(backend))
        index = self._normalize_retrieval_index(retrieval_index or self.retrieval_index)
        alpha = self.fusion_alpha if fusion_alpha is None else min(max(float(fusion_alpha), 0.0), 1.0)
        pool_size = self.fusion_pool_size if fusion_pool_size is None else max(int(fusion_pool_size), 1)
        top_k = self.result_top_k if result_top_k is None else max(int(result_top_k), 1)

        session_id = str(uuid.uuid4())
        session = RAIRSession(
            session_id=session_id,
            embedding_backend=backend,
            embedding_model=model_name,
            retrieval_index=index,
            fusion_alpha=alpha,
            fusion_pool_size=pool_size,
            result_top_k=top_k,
        )
        self._sessions[session_id] = session
        return {
            "session_id": session_id,
            "session": self._session_summary(session),
        }

    async def start_session(
        self,
        initial_query: str,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        retrieval_index: Optional[str] = None,
        fusion_alpha: Optional[float] = None,
        fusion_pool_size: Optional[int] = None,
        result_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        created = self.create_session(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            retrieval_index=retrieval_index,
            fusion_alpha=fusion_alpha,
            fusion_pool_size=fusion_pool_size,
            result_top_k=result_top_k,
        )
        return await self.submit_feedback(created["session_id"], initial_query)

    async def submit_feedback(self, session_id: str, message: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"RAIR session not found: {session_id}")

        message = str(message or "").strip()
        if not message:
            raise ValueError("message must not be empty")

        if session.turn == 0:
            session.initial_query = message
            session.current_query = message
        elif session.pending_suggestions:
            session.feedback_pairs.append(
                {
                    "turn": len(session.feedback_pairs) + 1,
                    "suggestions": list(session.pending_suggestions),
                    "answer": message,
                }
            )

        turn = await self._run_rair_turn(session=session, user_message=message)
        return {
            "session_id": session_id,
            "session": self._session_summary(session),
            "turn": turn,
        }
    def get_session(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"RAIR session not found: {session_id}")
        return self._session_summary(session, include_history=True)

    @staticmethod
    def _session_summary(
        session: RAIRSession,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session.session_id,
            "initial_query": session.initial_query,
            "current_query": session.current_query,
            "turn": session.turn,
            "embedding_backend": session.embedding_backend,
            "embedding_model": session.embedding_model,
            "retrieval_index": session.retrieval_index,
            "fusion_alpha": session.fusion_alpha if session.retrieval_index == "fusion" else None,
            "fusion_pool_size": session.fusion_pool_size if session.retrieval_index == "fusion" else None,
            "result_top_k": session.result_top_k,
            "feedback_pairs": session.feedback_pairs,
            "pending_suggestions": session.pending_suggestions,
        }
        if include_history:
            payload["history"] = session.history
        return payload
