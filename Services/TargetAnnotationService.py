from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import select

from Database.db_session import SessionLocal
from Entities.entities import VisDialTargetAnnotations


class TargetAnnotationService:
    """
    Read dialogue-derived visual facts for retrieved candidate images.

    These annotations enrich candidate evidence after retrieval. They should not
    be treated as the user's initial query or leaked as target-only information.
    """

    @staticmethod
    def _normalize_path(value: Any) -> str:
        return str(value or "").replace("\\", "/").lstrip("./").strip()

    @staticmethod
    def _basename(value: Any) -> str:
        return Path(str(value or "")).name

    @staticmethod
    def _to_payload(row: VisDialTargetAnnotations) -> Dict[str, Any]:
        return {
            "split": row.split,
            "dialog_index": row.dialog_index,
            "image_id": row.image_id,
            "image_path": row.image_path,
            "base_caption": row.base_caption,
            "visual_facts": row.visual_facts or [],
            "positive_facts": row.positive_facts or [],
            "negative_facts": row.negative_facts or [],
            "uncertain_facts": row.uncertain_facts or [],
            "enriched_caption": row.enriched_caption,
            "source": row.source,
        }

    def get_by_image_paths(self, image_paths: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        normalized_paths = [self._normalize_path(path) for path in image_paths if path]
        basenames = [self._basename(path) for path in normalized_paths]

        if not normalized_paths:
            return {}

        with SessionLocal() as session:
            rows = session.execute(
                select(VisDialTargetAnnotations).where(
                    VisDialTargetAnnotations.image_path.in_(normalized_paths)
                )
            ).scalars().all()

        by_path: Dict[str, Dict[str, Any]] = {}
        by_name: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            payload = self._to_payload(row)
            by_path[self._normalize_path(row.image_path)] = payload
            by_name[self._basename(row.image_path)] = payload

        result = {}
        for path in normalized_paths:
            result[path] = by_path.get(path) or by_name.get(self._basename(path)) or {}

        return result

    def get_by_image_id(self, image_id: Any, split: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with SessionLocal() as session:
            stmt = select(VisDialTargetAnnotations).where(
                VisDialTargetAnnotations.image_id == str(image_id)
            )
            if split:
                stmt = stmt.where(VisDialTargetAnnotations.split == split)

            row = session.execute(stmt).scalars().first()

        return self._to_payload(row) if row else None
