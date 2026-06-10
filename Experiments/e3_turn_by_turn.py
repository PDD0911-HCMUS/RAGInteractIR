import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Experiments.e1_baseline_retrieval import (
    build_rewrite_context,
    compute_gallery_overlap,
    compute_rank,
    load_annotations,
    metrics_from_rank,
    resolve_torch_device,
)
from Experiments.e2_rair_comparison import (
    caption_only_evidence,
    choose_oracle_suggestion,
    rank_change_metrics,
    score_oracle_suggestions,
)
from Services.QAFSService import QASF


logger = logging.getLogger("rair.e3")


def load_scenario(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        scenario = json.load(f)
    if "samples" not in scenario:
        raise ValueError(f"Invalid scenario file, missing samples: {path}")
    return scenario


def scenario_samples(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = []
    for item in scenario.get("samples", []):
        target_facts = item.get("target_facts") or []
        samples.append(
            {
                "split": item.get("split"),
                "dialog_index": item.get("dialog_index"),
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "base_caption": item.get("base_caption") or item.get("initial_query") or "",
                "enriched_caption": item.get("enriched_caption") or "",
                "visual_facts": target_facts,
                "positive_facts": item.get("positive_facts") or target_facts,
                "negative_facts": item.get("negative_facts") or [],
                "uncertain_facts": item.get("uncertain_facts") or [],
                "scenario_sample_id": item.get("sample_id"),
            }
        )
    return samples


def load_e3_samples(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if args.scenario:
        scenario = load_scenario(args.scenario)
        samples = scenario_samples(scenario)
        if args.offset:
            samples = samples[args.offset :]
        if args.limit is not None and args.limit >= 0:
            samples = samples[: args.limit]
        return samples, scenario

    return (
        load_annotations(
            split=args.split,
            limit=args.limit,
            offset=args.offset,
        ),
        None,
    )


def build_evidence(
    service: Any,
    method: str,
    query: str,
    image_ids: List[Any],
    captions: List[str],
    scores: List[float],
    evidence_top_k: int,
    fact_top_m: int,
    fact_alpha: float,
    fact_beta: float,
    fact_gamma: float,
    fact_delta: float,
) -> List[Dict[str, Any]]:
    if method == "rair_without_facts":
        return caption_only_evidence(
            image_ids=image_ids,
            captions=captions,
            scores=scores,
            top_k=evidence_top_k,
        )

    evidence = service.build_candidate_evidence(
        image_ids=image_ids,
        captions=captions,
        scores=scores,
        top_k=evidence_top_k,
    )

    if method == "rair_full_qafs":
        evidence = QASF(
            embedding_service=service,
            top_m=fact_top_m,
            alpha=fact_alpha,
            beta=fact_beta,
            gamma=fact_gamma,
            delta=fact_delta,
        ).select(query=query, evidence=evidence)

    return evidence


def strip_evidence_if_needed(turn_record: Dict[str, Any], save_evidence: bool) -> Dict[str, Any]:
    if save_evidence:
        return turn_record

    item = dict(turn_record)
    evidence = item.pop("candidate_evidence", None)
    if evidence is not None:
        item["candidate_evidence_count"] = len(evidence)
        item["candidate_evidence_top_ids"] = [
            candidate.get("image_id") for candidate in evidence[:5]
        ]
    return item


async def run_method_turns(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    method: str,
    initial_query: str,
    turns: int,
    ks: List[int],
    search_depth: int,
    evidence_top_k: int,
    oracle_overlap_threshold: int,
    fact_top_m: int,
    fact_alpha: float,
    fact_beta: float,
    fact_gamma: float,
    fact_delta: float,
    save_evidence: bool,
) -> Dict[str, Any]:
    current_query = initial_query
    current_ids, current_captions, current_scores = service.faiss_search(
        query_text=current_query,
        gallery=gallery,
        top_k=search_depth,
    )
    current_rank = compute_rank(current_ids, sample)
    turn_records = [
        {
            "turn": 0,
            "query": current_query,
            "rank_change": "initial",
            **metrics_from_rank(current_rank, ks),
            "top_ids": current_ids[: max(ks)],
            "top_scores": current_scores[: max(ks)],
            "top_captions": current_captions[: max(ks)],
        }
    ]

    for turn_id in range(1, turns + 1):
        evidence = build_evidence(
            service=service,
            method=method,
            query=current_query,
            image_ids=current_ids,
            captions=current_captions,
            scores=current_scores,
            evidence_top_k=evidence_top_k,
            fact_top_m=fact_top_m,
            fact_alpha=fact_alpha,
            fact_beta=fact_beta,
            fact_gamma=fact_gamma,
            fact_delta=fact_delta,
        )
        reasoning = await service.reasoning(
            history=None,
            input=current_query,
            candidate_evidence=evidence,
        )
        suggestions = score_oracle_suggestions(
            reasoning.get("suggestions", []),
            sample,
            current_query=current_query,
        )
        selected = choose_oracle_suggestion(
            suggestions,
            sample,
            current_query=current_query,
            min_overlap=oracle_overlap_threshold,
        )
        interaction_action = "accept" if selected else "no_op"
        if selected:
            next_query, compose_latency_ms = await service.compose_refined_query(
                current_query=current_query,
                accepted_suggestion=selected,
            )
        else:
            next_query = current_query
            compose_latency_ms = 0

        next_ids, next_captions, next_scores = service.faiss_search(
            query_text=next_query,
            gallery=gallery,
            top_k=search_depth,
        )
        next_rank = compute_rank(next_ids, sample)
        turn_record = {
            "turn": turn_id,
            "previous_query": current_query,
            "query": next_query,
            "interaction_action": interaction_action,
            "compose_latency_ms": compose_latency_ms,
            "query_refinement_policy": "llm_compose" if selected else "no_op",
            "selected_suggestion": selected,
            "suggestions": suggestions,
            "diagnosis": reasoning.get("diagnosis", {}),
            "candidate_evidence": evidence,
            "oracle_overlap_threshold": oracle_overlap_threshold,
            **rank_change_metrics(current_rank, next_rank),
            **metrics_from_rank(next_rank, ks),
            "top_ids": next_ids[: max(ks)],
            "top_scores": next_scores[: max(ks)],
            "top_captions": next_captions[: max(ks)],
        }
        turn_records.append(strip_evidence_if_needed(turn_record, save_evidence))

        current_query = next_query
        current_ids = next_ids
        current_captions = next_captions
        current_scores = next_scores
        current_rank = next_rank

    final_turn = turn_records[-1]
    return {
        "method": method,
        "initial_query": initial_query,
        "final_query": final_turn.get("query"),
        "turns": turn_records,
        "final": {
            key: final_turn.get(key)
            for key in ["rank", "mrr", *[f"hit@{k}" for k in ks]]
        },
    }


async def evaluate_sample(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    methods: List[str],
    turns: int,
    ks: List[int],
    search_depth: int,
    evidence_top_k: int,
    oracle_overlap_threshold: int,
    fact_top_m: int,
    fact_alpha: float,
    fact_beta: float,
    fact_gamma: float,
    fact_delta: float,
    save_evidence: bool,
) -> Dict[str, Any]:
    rewritten_query, _ = await service.rewrite_query(build_rewrite_context(sample))

    method_results = {}
    for method in methods:
        method_results[method] = await run_method_turns(
            service=service,
            gallery=gallery,
            sample=sample,
            method=method,
            initial_query=rewritten_query,
            turns=turns,
            ks=ks,
            search_depth=search_depth,
            evidence_top_k=evidence_top_k,
            oracle_overlap_threshold=oracle_overlap_threshold,
            fact_top_m=fact_top_m,
            fact_alpha=fact_alpha,
            fact_beta=fact_beta,
            fact_gamma=fact_gamma,
            fact_delta=fact_delta,
            save_evidence=save_evidence,
        )

    return {
        "split": sample.get("split"),
        "dialog_index": sample.get("dialog_index"),
        "image_id": sample.get("image_id"),
        "image_path": sample.get("image_path"),
        "scenario_sample_id": sample.get("scenario_sample_id"),
        "base_caption": sample.get("base_caption"),
        "rewritten_query": rewritten_query,
        "methods": method_results,
    }


def summarize_e3(results: List[Dict[str, Any]], methods: List[str], ks: List[int], turns: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(results),
        "methods": {},
    }

    for method in methods:
        method_summary = {
            "turns": {},
        }
        for turn_id in range(0, turns + 1):
            turn_items = []
            for result in results:
                method_result = result.get("methods", {}).get(method)
                if not method_result:
                    continue
                records = method_result.get("turns", [])
                if turn_id < len(records):
                    turn_items.append(records[turn_id])

            if not turn_items:
                continue

            ranks = [item["rank"] for item in turn_items if item.get("rank") is not None]
            turn_summary: Dict[str, Any] = {
                "num_samples": len(turn_items),
                "mrr": sum(item.get("mrr", 0.0) for item in turn_items) / len(turn_items),
                "mean_rank_found": (sum(ranks) / len(ranks)) if ranks else None,
                "found_rate": len(ranks) / len(turn_items),
            }
            for k in ks:
                key = f"hit@{k}"
                turn_summary[key] = sum(item.get(key, 0) for item in turn_items) / len(turn_items)

            interaction_items = [
                item for item in turn_items if item.get("turn", 0) > 0
            ]
            if interaction_items:
                accepted = sum(1 for item in interaction_items if item.get("interaction_action") == "accept")
                turn_summary["interaction"] = {
                    "accept_rate": accepted / len(interaction_items),
                    "no_op_rate": 1.0 - (accepted / len(interaction_items)),
                    "improved_rate": sum(item.get("improved", 0) for item in interaction_items) / len(interaction_items),
                    "worsened_rate": sum(item.get("worsened", 0) for item in interaction_items) / len(interaction_items),
                    "lost_target_rate": sum(item.get("lost_target", 0) for item in interaction_items) / len(interaction_items),
                    "recovered_target_rate": sum(item.get("recovered_target", 0) for item in interaction_items) / len(interaction_items),
                    "not_found_rate": (
                        sum(1 for item in interaction_items if item.get("rank_change") == "not_found")
                        / len(interaction_items)
                    ),
                }

            method_summary["turns"][str(turn_id)] = turn_summary

        method_summary["final"] = method_summary["turns"].get(str(turns), {})
        summary["methods"][method] = method_summary

    return summary


async def main_async(args: argparse.Namespace) -> None:
    methods = list(args.methods)
    ks = sorted(set(args.k))
    search_depth = max(args.search_depth, max(ks))
    samples, scenario = load_e3_samples(args)
    if not samples:
        raise ValueError("No E3 samples loaded")

    from Services.LocalLLMService import LocalLLMService
    from Services.OpenAIService import OpenAIService
    from Services.VisDialGPTCLIPService import VisDialGPTCLIPService

    if args.llm_provider == "local":
        llm_service = LocalLLMService(
            model_name=args.local_llm_model,
            device=args.local_llm_device,
            dtype=args.local_llm_dtype,
            local_files_only=not args.local_llm_online,
            max_new_tokens=args.local_llm_max_new_tokens,
        )
    else:
        llm_service = OpenAIService(model_name=args.reasoning_model)

    service = VisDialGPTCLIPService(
        vlm=args.clip_model,
        device=args.device,
        openai_service=llm_service,
        reasoning_model=args.local_llm_model if args.llm_provider == "local" else args.reasoning_model,
    )
    gallery = service.build_gallery()
    overlap = compute_gallery_overlap(samples, gallery)
    if overlap["overlap"] == 0:
        logger.warning("No target images were found in the retrieval gallery. Metrics will be 0.")

    results = []
    for index, sample in enumerate(samples, start=1):
        logger.info(
            "E3 sample %d/%d split=%s dialog_index=%s image_id=%s",
            index,
            len(samples),
            sample.get("split"),
            sample.get("dialog_index"),
            sample.get("image_id"),
        )
        try:
            result = await evaluate_sample(
                service=service,
                gallery=gallery,
                sample=sample,
                methods=methods,
                turns=args.turns,
                ks=ks,
                search_depth=search_depth,
                evidence_top_k=args.evidence_top_k,
                oracle_overlap_threshold=args.oracle_overlap_threshold,
                fact_top_m=args.fact_top_m,
                fact_alpha=args.fact_alpha,
                fact_beta=args.fact_beta,
                fact_gamma=args.fact_gamma,
                fact_delta=args.fact_delta,
                save_evidence=args.save_evidence,
            )
        except Exception:
            if not args.continue_on_error:
                raise
            logger.exception("Sample failed; recording error and continuing")
            result = {
                "split": sample.get("split"),
                "dialog_index": sample.get("dialog_index"),
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "base_caption": sample.get("base_caption"),
                "methods": {},
                "error": "sample_failed",
            }

        results.append(result)
        if args.output_jsonl:
            args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    payload = {
        "config": {
            "experiment": "E3",
            "scenario": str(args.scenario) if args.scenario else None,
            "scenario_name": scenario.get("scenario_name") if scenario else None,
            "split": args.split,
            "limit": args.limit,
            "offset": args.offset,
            "turns": args.turns,
            "methods": methods,
            "ks": ks,
            "search_depth": search_depth,
            "evidence_top_k": args.evidence_top_k,
            "oracle_overlap_threshold": args.oracle_overlap_threshold,
            "fact_selection": {
                "qafs_top_m": args.fact_top_m,
                "alpha": args.fact_alpha,
                "beta": args.fact_beta,
                "gamma": args.fact_gamma,
                "delta": args.fact_delta,
            },
            "clip_model": args.clip_model,
            "device": args.device,
            "llm_provider": args.llm_provider,
            "reasoning_model": args.reasoning_model,
            "local_llm_model": args.local_llm_model,
            "gallery_overlap": overlap,
            "interaction_policy": "novel_target_fact_overlap_accept_or_no_op",
        },
        "summary": summarize_e3(results, methods, ks, args.turns),
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E3 turn-by-turn RAIR evaluation over multiple interaction turns."
    )
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--evidence-top-k", type=int, default=3)
    parser.add_argument("--oracle-overlap-threshold", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rair_without_facts", "rair_full", "rair_full_qafs"],
        default=["rair_full_qafs"],
    )
    parser.add_argument("--fact-top-m", type=int, default=4)
    parser.add_argument("--fact-alpha", type=float, default=0.5)
    parser.add_argument("--fact-beta", type=float, default=0.3)
    parser.add_argument("--fact-gamma", type=float, default=0.2)
    parser.add_argument("--fact-delta", type=float, default=0.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=resolve_torch_device("CLIP_DEVICE"))
    parser.add_argument("--llm-provider", choices=["openai", "local"], default=os.environ.get("RAIR_LLM_PROVIDER", "openai"))
    parser.add_argument("--reasoning-model", default=os.environ.get("RAIR_REASONING_MODEL", "gpt-4o"))
    parser.add_argument("--local-llm-model", default=os.environ.get("RAIR_LOCAL_LLM_MODEL", "google/gemma-3-4b-it"))
    parser.add_argument("--local-llm-device", default=resolve_torch_device("LOCAL_LLM_DEVICE"))
    parser.add_argument("--local-llm-dtype", default=os.environ.get("LOCAL_LLM_DTYPE", "auto"))
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=768)
    parser.add_argument("--local-llm-online", action="store_true")
    parser.add_argument("--save-evidence", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("e3_turn_by_turn_results.json"))
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--continue-on-error", action="store_true")
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
