from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from sqlalchemy import text
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel

from Database.create_flickr30k_tables import (
    CREATE_FLICKR30K_CLIP_SQL,
    CREATE_FLICKR30K_SIGLIP_SQL,
    INDEX_SQL,
)
from Database.db_session import engine

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
except Exception:  # pragma: no cover
    Progress = None


SELECT_TARGETS_SQL = """
SELECT "image_id", "image_path", "base_caption", "captions"
FROM "Flickr30KTargetAnnotations"
WHERE "split" = :mode
ORDER BY "ID"
"""

INSERT_CLIP_SQL = """
INSERT INTO "Flickr30KCLIPCapDial"
    ("image_id", "caption", "captions", "dialog_id", "img_em", "cap_em", "mode", "image_path", "model_name")
VALUES
    (:image_id, :caption, CAST(:captions AS JSONB), :dialog_id, :img_em, :cap_em, :mode, :image_path, :model_name)
"""

INSERT_SIGLIP_SQL = """
INSERT INTO "Flickr30KSigLIPCapDial"
    ("image_id", "caption", "captions", "dialog_id", "img_em", "cap_em", "mode", "image_path", "model_name")
VALUES
    (:image_id, :caption, CAST(:captions AS JSONB), :dialog_id, :img_em, :cap_em, :mode, :image_path, :model_name)
"""

EXISTING_SQL = """
SELECT "image_id"
FROM {table_name}
WHERE "mode" = :mode AND "model_name" = :model_name
"""


def resolve_dtype(dtype: str) -> Optional[torch.dtype]:
    value = (dtype or "auto").lower()
    if value in {"auto", "none"}:
        return None
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def normalize_caption(base_caption: str, captions: Any, mode: str) -> str:
    caption_list = captions if isinstance(captions, list) else []
    if mode == "first":
        return base_caption or (caption_list[0] if caption_list else "")
    if mode == "all":
        values = [str(item).strip() for item in caption_list if str(item).strip()]
        return " ".join(values) if values else base_caption
    if mode == "base_plus_entities":
        return base_caption or (caption_list[0] if caption_list else "")
    raise ValueError(f"Unsupported caption mode: {mode}")


class Flickr30KEmbeddingBuilder:
    def __init__(
        self,
        image_folder: str,
        backend: str,
        model_name: str,
        mode: str,
        batchsize: int = 512,
        device: Optional[str] = None,
        dtype: str = "auto",
        caption_mode: str = "first",
    ) -> None:
        self.image_folder = Path(image_folder)
        self.backend = backend.lower()
        self.model_name = model_name
        self.mode = mode
        self.batchsize = batchsize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(dtype)
        self.caption_mode = caption_mode

        if self.backend not in {"clip", "siglip"}:
            raise ValueError("backend must be 'clip' or 'siglip'")

        print("[Flickr30KEmbeddings] Initializing...")
        print(f"[Flickr30KEmbeddings] backend       : {self.backend}")
        print(f"[Flickr30KEmbeddings] model         : {self.model_name}")
        print(f"[Flickr30KEmbeddings] mode          : {self.mode}")
        print(f"[Flickr30KEmbeddings] image_folder  : {self.image_folder}")
        print(f"[Flickr30KEmbeddings] caption_mode  : {self.caption_mode}")
        print(f"[Flickr30KEmbeddings] device        : {self.device}")

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.create_model()

    @property
    def table_name(self) -> str:
        return '"Flickr30KCLIPCapDial"' if self.backend == "clip" else '"Flickr30KSigLIPCapDial"'

    @property
    def insert_sql(self) -> str:
        return INSERT_CLIP_SQL if self.backend == "clip" else INSERT_SIGLIP_SQL

    def create_model(self) -> None:
        kwargs: Dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype

        if self.backend == "clip":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()
            return

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()

    @staticmethod
    def to_feature_tensor(output: Any, preferred_attr: str) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if hasattr(output, preferred_attr) and getattr(output, preferred_attr) is not None:
            return getattr(output, preferred_attr)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Unsupported feature output: {type(output)!r}")

    @torch.no_grad()
    def embed_image(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        feats = self.to_feature_tensor(output, "image_embeds")
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).detach().float().cpu().tolist()

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "clip":
            inputs = self.tokenizer(
                text=list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                text=list(texts),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

        output = self.model.get_text_features(**inputs)
        feats = self.to_feature_tensor(output, "text_embeds")
        feats = F.normalize(feats, dim=-1)
        return feats.detach().float().cpu().tolist()

    def ensure_table(self) -> None:
        with engine.begin() as conn:
            conn.execute(text(CREATE_FLICKR30K_CLIP_SQL if self.backend == "clip" else CREATE_FLICKR30K_SIGLIP_SQL))
            for sql in INDEX_SQL:
                if ("Flickr30KCLIPCapDial" in sql and self.backend == "clip") or (
                    "Flickr30KSigLIPCapDial" in sql and self.backend == "siglip"
                ):
                    conn.execute(text(sql))

    def load_existing_ids(self) -> Set[str]:
        sql = EXISTING_SQL.format(table_name=self.table_name)
        with engine.begin() as conn:
            rows = conn.execute(
                text(sql),
                {"mode": self.mode, "model_name": self.model_name},
            ).fetchall()
        return {str(row[0]) for row in rows if row[0] is not None}

    def load_targets(self, limit: Optional[int], offset: int) -> List[Tuple[str, str, str, Any]]:
        with engine.begin() as conn:
            rows = conn.execute(text(SELECT_TARGETS_SQL), {"mode": self.mode}).fetchall()
        targets = [(str(row[0]), str(row[1]), row[2] or "", row[3] or []) for row in rows]
        targets = targets[offset:]
        if limit is not None:
            targets = targets[:limit]
        return targets

    def build_embeddings(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        skip_existing: bool = False,
    ) -> int:
        self.ensure_table()
        targets = self.load_targets(limit=limit, offset=offset)
        existing = self.load_existing_ids() if skip_existing else set()
        buffer: List[Dict[str, Any]] = []
        inserted = 0

        def flush() -> None:
            nonlocal inserted, buffer
            if not buffer:
                return
            with engine.begin() as conn:
                conn.execute(text(self.insert_sql), buffer)
            inserted += len(buffer)
            buffer = []

        progress = None
        task = None
        if Progress is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            )
            progress.__enter__()
            task = progress.add_task(f"[green]Building Flickr30K {self.backend} embeddings...", total=len(targets))

        try:
            for image_id, image_path, base_caption, captions in targets:
                if image_id in existing:
                    if progress is not None:
                        progress.advance(task)
                    continue

                abs_image_path = self.image_folder / image_path
                try:
                    caption = normalize_caption(base_caption, captions, self.caption_mode)
                    img_em = self.embed_image(str(abs_image_path))
                    cap_em = self.embed_texts([caption])[0]
                    buffer.append(
                        {
                            "image_id": image_id,
                            "caption": caption,
                            "captions": json.dumps(captions, ensure_ascii=True),
                            "dialog_id": uuid.uuid4(),
                            "img_em": img_em,
                            "cap_em": cap_em,
                            "mode": self.mode,
                            "image_path": image_path,
                            "model_name": self.model_name,
                        }
                    )
                    existing.add(image_id)
                    if len(buffer) >= self.batchsize:
                        flush()
                        if progress is not None:
                            progress.console.print(f"[cyan]Inserted rows:[/cyan] {inserted}")
                except Exception as exc:
                    msg = f"[Flickr30KEmbeddings] Error image_id={image_id} path={image_path}: {exc}"
                    if progress is not None:
                        progress.console.print(f"[red]{msg}[/red]")
                    else:
                        print(msg)

                if progress is not None:
                    progress.advance(task)
        finally:
            flush()
            if progress is not None:
                progress.__exit__(None, None, None)

        print(f"Inserted Flickr30K {self.backend} rows: {inserted}")
        return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP/SigLIP embeddings for Flickr30K retrieval.")
    parser.add_argument("--image-folder", required=True, help="Root folder containing Flickr30K image paths stored in DB.")
    parser.add_argument("--backend", required=True, choices=["clip", "siglip"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--mode", default="train", help="Split/mode from Flickr30KTargetAnnotations.")
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--caption-mode", default="first", choices=["first", "all", "base_plus_entities"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name
    if model_name is None:
        model_name = "openai/clip-vit-base-patch32" if args.backend == "clip" else "google/siglip-base-patch16-224"
    builder = Flickr30KEmbeddingBuilder(
        image_folder=args.image_folder,
        backend=args.backend,
        model_name=model_name,
        mode=args.mode,
        batchsize=args.batchsize,
        device=args.device,
        dtype=args.dtype,
        caption_mode=args.caption_mode,
    )
    builder.build_embeddings(
        limit=args.limit,
        offset=args.offset,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
