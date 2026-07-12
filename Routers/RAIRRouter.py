import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Services.RAIRInteractiveRetrievalService import RAIRInteractiveRetrievalService


logger = logging.getLogger("rair.api")


class RAIRStartRequest(BaseModel):
    query: str = Field(..., min_length=1)
    embedding_backend: Optional[str] = Field(
        default=None,
        description="Retrieval embedding backend: siglip or clip.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Optional model override, e.g. openai/clip-vit-base-patch32.",
    )
    retrieval_index: Optional[str] = Field(
        default=None,
        description="image, caption, or fusion.",
    )
    fusion_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fusion weight for image score when retrieval_index=fusion.",
    )
    fusion_pool_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Candidate pool size for fusion reranking.",
    )


class RAIRFeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1)


class RAIRRouter:
    def __init__(self, service: RAIRInteractiveRetrievalService) -> None:
        self.service = service
        self.router = APIRouter(prefix="/rair", tags=["RAIR"])
        self.router.add_api_route(
            "/sessions",
            self.start_session,
            methods=["POST"],
            summary="Start a RAIR-VF retrieval session",
        )
        self.router.add_api_route(
            "/sessions/{session_id}/turns",
            self.submit_feedback,
            methods=["POST"],
            summary="Submit user feedback and run the next RAIR-VF turn",
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.get_session,
            methods=["GET"],
            summary="Get RAIR-VF session state",
        )

    def preload_gallery(self) -> None:
        self.service.preload()

    async def start_session(self, req: RAIRStartRequest) -> Dict[str, Any]:
        try:
            return await self.service.start_session(
                initial_query=req.query,
                embedding_backend=req.embedding_backend,
                embedding_model=req.embedding_model,
                retrieval_index=req.retrieval_index,
                fusion_alpha=req.fusion_alpha,
                fusion_pool_size=req.fusion_pool_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RAIR session start failed")
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

    async def submit_feedback(
        self,
        session_id: str,
        req: RAIRFeedbackRequest,
    ) -> Dict[str, Any]:
        try:
            return await self.service.submit_feedback(session_id, req.message)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RAIR feedback turn failed session_id=%s", session_id)
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        try:
            return self.service.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
