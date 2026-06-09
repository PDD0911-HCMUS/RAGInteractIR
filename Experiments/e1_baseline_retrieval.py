import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Database.db_session import SessionLocal
from Entities.entities import VisDialTargetAnnotations


logger = logging.getLogger("rair.e1")


def resolve_torch_device(env_name: str, default: str = "cpu") -> str:
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


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./").strip()


def coco_id_from_path(value: Any) -> Optional[int]:
    stem = Path(str(value or "")).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def is_groundtruth_match(result_id: Any, target: Dict[str, Any]) -> bool:
    result_path = normalize_path(result_id)
    target_path = normalize_path(target.get("image_path"))

    if result_path == target_path:
        return True

    if Path(result_path).name == Path(target_path).name:
        return True

    result_coco_id = coco_id_from_path(result_path)
    target_image_id = target.get("image_id")
    try:
        target_image_id = int(target_image_id)
    except (TypeError, ValueError):
        target_image_id = coco_id_from_path(target_path)

    return result_coco_id is not None and result_coco_id == target_image_id


def load_annotations(split: str, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(VisDialTargetAnnotations)
            .where(VisDialTargetAnnotations.split == split)
            .order_by(VisDialTargetAnnotations.dialog_index)
            .offset(offset)
        )
        if limit is not None and limit >= 0:
            stmt = stmt.limit(limit)

        rows = session.execute(stmt).scalars().all()

    return [
        {
            "split": row.split,
            "dialog_index": row.dialog_index,
            "image_id": row.image_id,
            "image_path": row.image_path,
            "base_caption": row.base_caption,
            "enriched_caption": row.enriched_caption,
            "visual_facts": row.visual_facts or [],
            "positive_facts": row.positive_facts or [],
            "negative_facts": row.negative_facts or [],
            "uncertain_facts": row.uncertain_facts or [],
        }
        for row in rows
    ]


def compute_rank(retrieved_ids: Iterable[Any], target: Dict[str, Any]) -> Optional[int]:
    for rank, result_id in enumerate(retrieved_ids, start=1):
        if is_groundtruth_match(result_id, target):
            return rank
    return None


def metrics_from_rank(rank: Optional[int], ks: List[int]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }
    for k in ks:
        metrics[f"hit@{k}"] = int(rank is not None and rank <= k)
    return metrics


def summarize(results: List[Dict[str, Any]], methods: List[str], ks: List[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(results),
        "methods": {},
    }

    for method in methods:
        items = [
            result["methods"][method]
            for result in results
            if method in result.get("methods", {})
        ]
        if not items:
            continue

        ranks = [item["rank"] for item in items if item.get("rank") is not None]
        method_summary: Dict[str, Any] = {
            "num_samples": len(items),
            "mrr": sum(item.get("mrr", 0.0) for item in items) / len(items),
            "mean_rank_found": (sum(ranks) / len(ranks)) if ranks else None,
            "found_rate": len(ranks) / len(items),
        }
        for k in ks:
            key = f"hit@{k}"
            method_summary[key] = sum(item.get(key, 0) for item in items) / len(items)

        summary["methods"][method] = method_summary

    return summary


def compute_gallery_overlap(samples: List[Dict[str, Any]], gallery: Dict[str, Any]) -> Dict[str, Any]:
    gallery_ids = set()
    for image_id in gallery.get("image_id", []):
        normalized = normalize_path(image_id)
        if normalized:
            gallery_ids.add(normalized)
        coco_id = coco_id_from_path(normalized)
        if coco_id is not None:
            gallery_ids.add(str(coco_id))

    overlap = 0
    for sample in samples:
        target_keys = {
            normalize_path(sample.get("image_path")),
            str(sample.get("image_id")),
        }
        target_path_id = coco_id_from_path(sample.get("image_path"))
        if target_path_id is not None:
            target_keys.add(str(target_path_id))

        if any(key and key in gallery_ids for key in target_keys):
            overlap += 1

    total = len(samples)
    return {
        "num_samples": total,
        "overlap": overlap,
        "overlap_rate": overlap / total if total else 0.0,
    }


def build_rewrite_context(sample: Dict[str, Any]) -> Dict[str, Any]:
    caption = sample.get("base_caption") or ""
    return {
        "initial_query": caption,
        "feedback_pairs": [],
        "pending_suggestions": [],
        "latest_user_message": caption,
    }


async def evaluate_sample(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    methods: List[str],
    ks: List[int],
    search_depth: int,
) -> Dict[str, Any]:
    queries: Dict[str, str] = {}

    if "caption_clip" in methods:
        queries["caption_clip"] = sample.get("base_caption") or ""

    if "enriched_upper_bound" in methods:
        queries["enriched_upper_bound"] = (
            sample.get("enriched_caption")
            or sample.get("base_caption")
            or ""
        )

    if "rewrite_only" in methods:
        rewritten_query, _ = await service.rewrite_query(build_rewrite_context(sample))
        queries["rewrite_only"] = rewritten_query

    method_results: Dict[str, Any] = {}
    for method, query in queries.items():
        if not query.strip():
            continue

        retrieved_ids, captions, scores = service.faiss_search(
            query_text=query,
            gallery=gallery,
            top_k=search_depth,
        )
        rank = compute_rank(retrieved_ids, sample)
        method_results[method] = {
            "query": query,
            **metrics_from_rank(rank, ks),
            "top_ids": retrieved_ids[: max(ks)],
            "top_scores": scores[: max(ks)],
            "top_captions": captions[: max(ks)],
        }

    return {
        "split": sample.get("split"),
        "dialog_index": sample.get("dialog_index"),
        "image_id": sample.get("image_id"),
        "image_path": sample.get("image_path"),
        "base_caption": sample.get("base_caption"),
        "methods": method_results,
    }


async def main_async(args: argparse.Namespace) -> None:
    methods = list(args.methods)
    ks = sorted(set(args.k))
    search_depth = max(args.search_depth, max(ks))

    samples = load_annotations(
        split=args.split,
        limit=args.limit,
        offset=args.offset,
    )
    if not samples:
        raise ValueError(f"No VisDialTargetAnnotations rows loaded for split={args.split}")

    from Services.LocalLLMService import LocalLLMService
    from Services.OpenAIService import OpenAIService
    from Services.VisDialGPTCLIPService import VisDialGPTCLIPService

    llm_service = None
    if "rewrite_only" in methods:
        if args.llm_provider == "local":
            llm_service = LocalLLMService(
                model_name=args.local_llm_model,
                device=args.local_llm_device,
                dtype=args.local_llm_dtype,
                local_files_only=not args.local_llm_online,
                max_new_tokens=args.local_llm_max_new_tokens,
            )
        else:
            llm_service = OpenAIService(model_name=args.rewrite_model)

    service = VisDialGPTCLIPService(
        vlm=args.clip_model,
        device=args.device,
        openai_service=llm_service,
        reasoning_model=args.local_llm_model if args.llm_provider == "local" else args.rewrite_model,
    )
    gallery = service.build_gallery()
    overlap = compute_gallery_overlap(samples, gallery)
    if overlap["overlap"] == 0:
        logger.warning(
            "No target images from split=%s were found in the retrieval gallery. "
            "All Hit@K metrics will be 0. Build a gallery for this split or use a split that overlaps the gallery.",
            args.split,
        )
    else:
        logger.info(
            "Gallery overlap for split=%s: %d/%d (%.2f%%)",
            args.split,
            overlap["overlap"],
            overlap["num_samples"],
            overlap["overlap_rate"] * 100,
        )

    results = []
    for index, sample in enumerate(samples, start=1):
        logger.info(
            "E1 sample %d/%d split=%s dialog_index=%s image_id=%s",
            index,
            len(samples),
            args.split,
            sample.get("dialog_index"),
            sample.get("image_id"),
        )
        result = await evaluate_sample(
            service=service,
            gallery=gallery,
            sample=sample,
            methods=methods,
            ks=ks,
            search_depth=search_depth,
        )
        results.append(result)

        if args.output_jsonl:
            args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    payload = {
        "config": {
            "split": args.split,
            "limit": args.limit,
            "offset": args.offset,
            "methods": methods,
            "ks": ks,
            "search_depth": search_depth,
            "clip_model": args.clip_model,
            "device": args.device,
            "llm_provider": args.llm_provider,
            "rewrite_model": args.rewrite_model,
            "local_llm_model": args.local_llm_model,
            "gallery_overlap": overlap,
        },
        "summary": summarize(results, methods, ks),
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1 baseline retrieval: Caption CLIP vs Rewrite-only vs Enriched upper bound."
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate. Omit or use -1 to run the full split.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["caption_clip", "rewrite_only", "enriched_upper_bound"],
        default=["caption_clip", "rewrite_only", "enriched_upper_bound"],
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=resolve_torch_device("CLIP_DEVICE"))
    parser.add_argument("--llm-provider", choices=["openai", "local"], default=os.environ.get("RAIR_LLM_PROVIDER", "openai"))
    parser.add_argument("--rewrite-model", default=os.environ.get("RAIR_REWRITE_MODEL", "gpt-5.4"))
    parser.add_argument("--local-llm-model", default=os.environ.get("RAIR_LOCAL_LLM_MODEL", "google/gemma-3-4b-it"))
    parser.add_argument("--local-llm-device", default=resolve_torch_device("LOCAL_LLM_DEVICE"))
    parser.add_argument("--local-llm-dtype", default=os.environ.get("LOCAL_LLM_DTYPE", "auto"))
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--local-llm-online",
        action="store_true",
        help="Allow HuggingFace downloads/lookups for the local LLM if it is not cached.",
    )
    parser.add_argument("--output", type=Path, default=Path("e1_baseline_retrieval_results.json"))
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        args.limit = None
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
