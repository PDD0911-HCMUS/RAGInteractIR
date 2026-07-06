from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from PIL import Image
from sqlalchemy import text
from transformers import AutoModel, AutoProcessor

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - rich is optional for long-running scripts
    Progress = None

from Database.db_session import engine


CREATE_SIGLIP_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS "VisDialSigLIPCapDial" (
    "ID" BIGSERIAL PRIMARY KEY,
    "image_id" TEXT,
    "caption" TEXT,
    "dialog_id" UUID,
    "img_em" DOUBLE PRECISION[],
    "cap_em" DOUBLE PRECISION[],
    "mode" TEXT,
    "image_path" TEXT,
    "model_name" TEXT,
    "created_at" TIMESTAMPTZ DEFAULT NOW()
);
'''

INSERT_SIGLIP_SQL = '''
INSERT INTO "VisDialSigLIPCapDial"
    ("image_id", "caption", "dialog_id", "img_em", "cap_em", "mode", "image_path", "model_name")
VALUES
    (:image_id, :caption, :dialog_id, :img_em, :cap_em, :mode, :image_path, :model_name)
'''

EXISTING_IDS_SQL = '''
SELECT "image_id"
FROM "VisDialSigLIPCapDial"
WHERE "mode" = :mode AND "model_name" = :model_name
'''


class VisDialSigLIP:
    """
    Build SigLIP image/caption embeddings for VisDial retrieval.

    This mirrors datasets/VisDialCLIP.py but writes to VisDialSigLIPCapDial so
    retrieval services can switch the embedding backbone without changing the
    original CLIP table.
    """

    def __init__(
        self,
        map_image: str,
        image_folder: str,
        anno: str,
        model_name: str,
        mode: str,
        batchsize: int = 512,
        device: Optional[str] = None,
        dtype: str = "auto",
    ) -> None:
        print("[VisDialSigLIP] Initializing dataset...")
        print(f"[VisDialSigLIP] anno file      : {anno}")
        print(f"[VisDialSigLIP] map_image file : {map_image}")
        print(f"[VisDialSigLIP] image_folder   : {image_folder}")
        print(f"[VisDialSigLIP] model          : {model_name}")
        print(f"[VisDialSigLIP] mode           : {mode}")
        print(f"[VisDialSigLIP] batchsize      : {batchsize}")

        with open(anno, "r", encoding="utf-8") as f:
            self.ann = json.load(f)
        with open(map_image, "r", encoding="utf-8") as f:
            self.map = json.load(f)

        self.image_folder = Path(image_folder)
        self.dialogs = self.ann["data"]["dialogs"]
        self.model_name = model_name
        self.mode = mode
        self.batchsize = batchsize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype(dtype)

        self.processor, self.model = self.create_model()

        print(f"[VisDialSigLIP] Number of dialogs : {len(self.dialogs)}")
        print(f"[VisDialSigLIP] Number of map items: {len(self.map)}")
        print(f"[VisDialSigLIP] Device selected    : {self.device}")

    @staticmethod
    def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
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

    def create_model(self):
        processor = AutoProcessor.from_pretrained(self.model_name)
        kwargs: Dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()
        return processor, model

    @staticmethod
    def _to_feature_tensor(output: Any, preferred_attr: str) -> torch.Tensor:
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
        raise TypeError(f"Unsupported SigLIP feature output: {type(output)!r}")

    @torch.no_grad()
    def embed_image(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        feats = self._to_feature_tensor(output, "image_embeds")
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).detach().float().cpu().tolist()

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        output = self.model.get_text_features(**inputs)
        feats = self._to_feature_tensor(output, "text_embeds")
        feats = F.normalize(feats, dim=-1)
        return feats.detach().float().cpu().tolist()

    def ensure_table(self) -> None:
        with engine.begin() as conn:
            conn.execute(text(CREATE_SIGLIP_TABLE_SQL))

    def load_existing_image_ids(self) -> Set[str]:
        with engine.begin() as conn:
            rows = conn.execute(
                text(EXISTING_IDS_SQL),
                {"mode": self.mode, "model_name": self.model_name},
            ).fetchall()
        return {str(row[0]) for row in rows if row[0] is not None}

    def _dialog_iter(self, limit: Optional[int] = None, offset: int = 0) -> Iterable[Dict[str, Any]]:
        dialogs = self.dialogs[offset:]
        if limit is not None:
            dialogs = dialogs[:limit]
        return dialogs

    def build_capdial_embeddings(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        skip_existing: bool = False,
    ) -> int:
        self.ensure_table()
        existing = self.load_existing_image_ids() if skip_existing else set()
        dialogs = list(self._dialog_iter(limit=limit, offset=offset))
        inserted_rows = 0
        buffer: List[Dict[str, Any]] = []

        progress_ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            )
            if Progress is not None
            else None
        )

        def flush() -> None:
            nonlocal inserted_rows, buffer
            if not buffer:
                return
            with engine.begin() as conn:
                conn.execute(text(INSERT_SIGLIP_SQL), buffer)
            inserted_rows += len(buffer)
            buffer = []

        if progress_ctx is None:
            iterator = enumerate(dialogs, start=1)
            task = None
        else:
            progress_ctx.__enter__()
            task = progress_ctx.add_task("[green]Building SigLIP CapDial embeddings...", total=len(dialogs))
            iterator = enumerate(dialogs, start=1)

        try:
            for _, item in iterator:
                image_id = str(item["image_id"])
                if image_id in existing:
                    if progress_ctx is not None:
                        progress_ctx.advance(task)
                    continue

                image_path = self.map[image_id]
                abs_image_path = self.image_folder / image_path
                try:
                    img_em = self.embed_image(str(abs_image_path))
                    cap_em = self.embed_texts([item["caption"]])[0]
                    buffer.append(
                        {
                            "image_id": image_id,
                            "caption": item["caption"],
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
                        if progress_ctx is not None:
                            progress_ctx.console.print(f"[cyan]Inserted rows:[/cyan] {inserted_rows}")
                except Exception as exc:
                    msg = f"[VisDialSigLIP] Error image_id={image_id} path={image_path}: {exc}"
                    if progress_ctx is not None:
                        progress_ctx.console.print(f"[red]{msg}[/red]")
                    else:
                        print(msg)

                if progress_ctx is not None:
                    progress_ctx.advance(task)
        finally:
            flush()
            if progress_ctx is not None:
                progress_ctx.__exit__(None, None, None)

        print("\n================================")
        print("Done inserting SigLIP CapDial embeddings")
        print("Total rows inserted:", inserted_rows)
        print("================================")
        return inserted_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigLIP embeddings for VisDial image/caption retrieval.")
    parser.add_argument("--map-image", required=True, help="Path to coco2014_id_to_relpath.json")
    parser.add_argument("--anno", required=True, help="Path to VisDial annotation JSON")
    parser.add_argument("--image-folder", required=True, help="Root folder containing MSCOCO image paths")
    parser.add_argument("--model-name", default="google/siglip-base-patch16-224")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default: auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = VisDialSigLIP(
        map_image=args.map_image,
        anno=args.anno,
        image_folder=args.image_folder,
        model_name=args.model_name,
        mode=args.mode,
        batchsize=args.batchsize,
        device=args.device,
        dtype=args.dtype,
    )
    builder.build_capdial_embeddings(
        limit=args.limit,
        offset=args.offset,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
