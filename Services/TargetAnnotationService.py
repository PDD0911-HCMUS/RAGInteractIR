from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import select, text

from Database.db_session import SessionLocal
from Entities.entities import VisDialTargetAnnotations


class TargetAnnotationService:
    """
    Read visual facts for retrieved candidate images.

    VisDial uses dialogue-derived facts, while Flickr30K uses phrase-region
    annotations from Flickr30K Entities. These annotations enrich candidate
    evidence after retrieval and should not be treated as target-only user input.
    """

    def __init__(self, dataset: str = "visdial") -> None:
        value = str(dataset or "visdial").strip().lower()
        aliases = {
            "visdial": "visdial",
            "visdial2014": "visdial",
            "flickr": "flickr30k",
            "flickr30k": "flickr30k",
            "flickr30k_entities": "flickr30k",
        }
        if value not in aliases:
            raise ValueError(f"Unsupported annotation dataset: {dataset}")
        self.dataset = aliases[value]

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

    @staticmethod
    def _flickr_payload(row: Any) -> Dict[str, Any]:
        return {
            "split": row.split,
            "dialog_index": None,
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

        if self.dataset == "flickr30k":
            with SessionLocal() as session:
                rows = session.execute(
                    text(
                        '''
                        SELECT "split", "image_id", "image_path", "base_caption",
                               "visual_facts", "positive_facts", "negative_facts",
                               "uncertain_facts", "enriched_caption", "source"
                        FROM "Flickr30KTargetAnnotations"
                        WHERE "image_path" = ANY(:paths) OR "image_id" = ANY(:image_ids)
                        '''
                    ),
                    {
                        "paths": normalized_paths,
                        "image_ids": [Path(name).stem for name in basenames],
                    },
                ).all()

            by_path: Dict[str, Dict[str, Any]] = {}
            by_name: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                payload = self._flickr_payload(row)
                by_path[self._normalize_path(row.image_path)] = payload
                by_name[self._basename(row.image_path)] = payload
            return {
                path: by_path.get(path) or by_name.get(self._basename(path)) or {}
                for path in normalized_paths
            }

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

        return {
            path: by_path.get(path) or by_name.get(self._basename(path)) or {}
            for path in normalized_paths
        }

    def get_by_image_id(self, image_id: Any, split: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.dataset == "flickr30k":
            with SessionLocal() as session:
                sql = '''
                    SELECT "split", "image_id", "image_path", "base_caption",
                           "visual_facts", "positive_facts", "negative_facts",
                           "uncertain_facts", "enriched_caption", "source"
                    FROM "Flickr30KTargetAnnotations"
                    WHERE "image_id" = :image_id
                '''
                params = {"image_id": str(image_id)}
                if split:
                    sql += ' AND "split" = :split'
                    params["split"] = split
                row = session.execute(text(sql), params).first()
            return self._flickr_payload(row) if row else None

        with SessionLocal() as session:
            stmt = select(VisDialTargetAnnotations).where(
                VisDialTargetAnnotations.image_id == str(image_id)
            )
            if split:
                stmt = stmt.where(VisDialTargetAnnotations.split == split)

            row = session.execute(stmt).scalars().first()

        return self._to_payload(row) if row else None
