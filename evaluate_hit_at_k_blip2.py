import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from Routers.VLMChatRouter import MessageRequest, VLMChatRouter
from Services.BLIP2SimulateService import BLIP2SimulateService
from Storage.ConversationStore import ConversationStore


logger = logging.getLogger("rair.eval")


DEFAULT_GROUNDTRUTH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "VisDial"
    / "visdial_train_groundtruth_queries.json"
)


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./").strip()


def coco_id_from_path(value: Any) -> Optional[int]:
    stem = Path(str(value or "")).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def is_groundtruth_match(result_id: Any, groundtruth: Dict[str, Any]) -> bool:
    result_path = normalize_path(result_id)
    target_path = normalize_path(groundtruth.get("image_path"))

    if result_path == target_path:
        return True

    if Path(result_path).name == Path(target_path).name:
        return True

    result_coco_id = coco_id_from_path(result_path)
    target_image_id = groundtruth.get("image_id")
    try:
        target_image_id = int(target_image_id)
    except (TypeError, ValueError):
        target_image_id = coco_id_from_path(target_path)

    return result_coco_id is not None and result_coco_id == target_image_id


def hit_at_k(retrieved_ids: Iterable[Any], groundtruth: Dict[str, Any], k: int) -> int:
    return int(any(is_groundtruth_match(item, groundtruth) for item in list(retrieved_ids)[:k]))


def compute_hits(retrieved_ids: Iterable[Any], groundtruth: Dict[str, Any], ks: List[int]) -> Dict[str, int]:
    ids = list(retrieved_ids)
    return {f"hit@{k}": hit_at_k(ids, groundtruth, k) for k in ks}


def load_groundtruth(path: Path, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if offset:
        data = data[offset:]
    if limit is not None:
        data = data[:limit]

    return data


async def run_rair_turn(
    router: VLMChatRouter,
    conversation_id: str,
    message: str,
) -> Dict[str, Any]:
    return await router.send_message(
        conversation_id=conversation_id,
        req=MessageRequest(message=message),
    )


async def evaluate_sample(
    router: VLMChatRouter,
    simulator: BLIP2SimulateService,
    sample: Dict[str, Any],
    ks: List[int],
    turns: int,
    use_blip2: bool,
    early_stop_k: Optional[int],
) -> Dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    router.store.create(conversation_id=conversation_id)

    user_message = str(sample.get("query") or "").strip()
    if not user_message:
        raise ValueError(f"Missing query for sample: {sample}")

    sample_result = {
        "conversation_id": conversation_id,
        "image_id": sample.get("image_id"),
        "image_path": sample.get("image_path"),
        "initial_query": user_message,
        "turns": [],
    }

    for turn_idx in range(turns):
        logger.info(
            "Evaluating image_id=%s turn=%d message=%s",
            sample.get("image_id"),
            turn_idx,
            user_message,
        )
        response = await run_rair_turn(router, conversation_id, user_message)
        retrieved_ids = response.get("retrieve", {}).get("id", [])
        hits = compute_hits(retrieved_ids, sample, ks)

        turn_result = {
            "turn": turn_idx,
            "user_message": user_message,
            "rewritten_query": response.get("rewritten_query"),
            "triplets": response.get("triplets", []),
            "hits": hits,
            "early_stopped": False,
            "top_ids": retrieved_ids[: max(ks)],
            "suggestions": response.get("pending_suggestions", []),
            "simulated_answer": None,
        }

        logger.info(
            "Hit results image_id=%s turn=%d %s",
            sample.get("image_id"),
            turn_idx,
            hits,
        )

        should_stop = (
            early_stop_k is not None
            and hits.get(f"hit@{early_stop_k}", 0) == 1
        )
        if should_stop:
            turn_result["early_stopped"] = True
            sample_result["turns"].append(turn_result)
            logger.info(
                "Early stop image_id=%s turn=%d hit@%d=1",
                sample.get("image_id"),
                turn_idx,
                early_stop_k,
            )
            break

        if turn_idx < turns - 1:
            if not use_blip2:
                break

            feedback = simulator.simulate_feedback(
                target_image_path=sample.get("image_path"),
                suggestions=response.get("pending_suggestions", []),
                context_state=response.get("context_state", {}),
            )
            user_message = feedback["answer"]
            turn_result["simulated_answer"] = user_message
            logger.info("Simulated user answer: %s", user_message)

        sample_result["turns"].append(turn_result)

    return sample_result


def summarize(results: List[Dict[str, Any]], ks: List[int], turns: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(results),
        "ks": ks,
        "turns": {},
    }

    for turn_idx in range(turns):
        turn_items = [
            sample["turns"][turn_idx]
            for sample in results
            if len(sample.get("turns", [])) > turn_idx
        ]
        if not turn_items:
            continue

        summary["turns"][str(turn_idx)] = {
            f"hit@{k}": sum(item["hits"][f"hit@{k}"] for item in turn_items) / len(turn_items)
            for k in ks
        }
        summary["turns"][str(turn_idx)]["num_samples"] = len(turn_items)

    return summary


async def main_async(args: argparse.Namespace) -> None:
    groundtruth = load_groundtruth(
        path=args.groundtruth,
        limit=args.limit,
        offset=args.offset,
    )
    if not groundtruth:
        raise ValueError("No groundtruth samples loaded.")

    store = ConversationStore()
    router = VLMChatRouter(conversation_store=store)
    simulator = BLIP2SimulateService(
        model_name=args.blip2_model,
        device=args.blip2_device,
        local_files_only=args.hf_local_files_only,
        image_root=args.image_root,
        max_new_tokens=args.max_new_tokens,
    )

    results = []
    for idx, sample in enumerate(groundtruth, start=1):
        logger.info(
            "Starting sample %d/%d image_id=%s",
            idx,
            len(groundtruth),
            sample.get("image_id"),
        )
        result = await evaluate_sample(
            router=router,
            simulator=simulator,
            sample=sample,
            ks=args.k,
            turns=args.turns,
            use_blip2=not args.no_blip2,
            early_stop_k=args.early_stop_k,
        )
        results.append(result)

        if args.output_jsonl:
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize(results, args.k, args.turns)
    payload = {
        "summary": summary,
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAIR Hit@K with BLIP-2 simulated user feedback."
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        default=DEFAULT_GROUNDTRUTH,
        help="JSON file with image_id, image_path, query.",
    )
    parser.add_argument("--output", type=Path, default=Path("hit_at_k_blip2_results.json"))
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument(
        "--early-stop-k",
        type=int,
        default=20,
        help="Stop a sample once hit@K is 1. Use --early-stop-k 0 to disable.",
    )
    parser.add_argument(
        "--no-blip2",
        action="store_true",
        help="Run only the initial RAIR turns available without BLIP-2 feedback.",
    )
    parser.add_argument("--blip2-model", default=None)
    parser.add_argument("--blip2-device", default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--hf-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only local Hugging Face cache for BLIP-2.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.early_stop_k == 0:
        args.early_stop_k = None
    elif args.early_stop_k not in args.k:
        args.k = sorted(set(args.k + [args.early_stop_k]))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
