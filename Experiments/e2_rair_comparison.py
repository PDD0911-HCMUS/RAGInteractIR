import argparse
import asyncio
import json
import logging
import os
import re
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
    summarize,
)


logger = logging.getLogger("rair.e2")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with",
    "is", "are", "was", "were", "be", "being", "been", "this", "that",
    "there", "it", "he", "she", "they", "them", "his", "her", "their",
    "image", "photo", "picture", "visual", "facts", "additional",
}

CONTRADICTION_PAIRS = [
    ("inside", "outside"),
    ("indoor", "outdoor"),
    ("indoors", "outdoors"),
    ("day", "night"),
    ("black", "white"),
    ("standing", "sitting"),
    ("sitting", "lying"),
    ("laying", "standing"),
    ("grass", "floor"),
]


def summarize_e2(results: List[Dict[str, Any]], methods: List[str], ks: List[int]) -> Dict[str, Any]:
    summary = summarize(results, methods, ks)

    for method in methods:
        items = [
            result["methods"][method]
            for result in results
            if method in result.get("methods", {})
        ]
        if not items:
            continue

        interaction_items = [
            item for item in items if "interaction_action" in item
        ]
        if not interaction_items:
            continue

        count = len(interaction_items)
        accepted = sum(1 for item in interaction_items if item.get("interaction_action") == "accept")
        changes = {
            "improved": sum(item.get("improved", 0) for item in interaction_items),
            "worsened": sum(item.get("worsened", 0) for item in interaction_items),
            "unchanged": sum(item.get("unchanged", 0) for item in interaction_items),
            "lost_target": sum(item.get("lost_target", 0) for item in interaction_items),
            "recovered_target": sum(item.get("recovered_target", 0) for item in interaction_items),
            "not_found": sum(1 for item in interaction_items if item.get("rank_change") == "not_found"),
        }
        overlaps = [
            item.get("selected_suggestion").get("oracle_overlap")
            for item in interaction_items
            if item.get("selected_suggestion")
        ]

        summary["methods"][method]["interaction"] = {
            "accept_rate": accepted / count,
            "no_op_rate": 1.0 - (accepted / count),
            "mean_selected_overlap": (
                sum(overlaps) / len(overlaps) if overlaps else None
            ),
            **{f"{key}_rate": value / count for key, value in changes.items()},
            **changes,
        }

    return summary


def tokenize(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def oracle_tokens(sample: Dict[str, Any]) -> set[str]:
    parts = [
        sample.get("base_caption", ""),
        sample.get("enriched_caption", ""),
        " ".join(sample.get("visual_facts") or []),
        " ".join(sample.get("positive_facts") or []),
        " ".join(sample.get("negative_facts") or []),
    ]
    return tokenize(" ".join(parts))


def choose_oracle_suggestion(
    suggestions: List[Dict[str, Any]],
    sample: Dict[str, Any],
    min_overlap: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Simulate suggestion-based interaction with target-side visual facts. The
    oracle accepts the best supported suggestion only when it reaches the
    configured overlap threshold; otherwise it rejects all suggestions.
    """
    target_tokens = oracle_tokens(sample)
    if not suggestions or not target_tokens:
        return None

    best = None
    best_score = -1
    for suggestion in suggestions:
        text = " ".join(
            [
                str(suggestion.get("sug", "")),
                str(suggestion.get("explain", "")),
                str(suggestion.get("type", "")),
            ]
        )
        score = len(tokenize(text) & target_tokens)
        if score > best_score:
            best = suggestion
            best_score = score

    if best is None:
        return None

    selected = dict(best)
    selected["oracle_overlap"] = best_score
    selected["oracle_target_tokens"] = sorted(tokenize(
        " ".join(
            [
                str(best.get("sug", "")),
                str(best.get("explain", "")),
                str(best.get("type", "")),
            ]
        )
    ) & target_tokens)

    if best_score < min_overlap:
        selected["oracle_action"] = "reject"
        selected["oracle_reason"] = (
            f"best suggestion overlap {best_score} is below threshold {min_overlap}"
        )
        return None

    selected["oracle_action"] = "accept"
    selected["oracle_reason"] = (
        f"best suggestion overlap {best_score} meets threshold {min_overlap}"
    )
    return selected


def score_oracle_suggestions(
    suggestions: List[Dict[str, Any]],
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target_tokens = oracle_tokens(sample)
    scored = []
    for suggestion in suggestions:
        text = " ".join(
            [
                str(suggestion.get("sug", "")),
                str(suggestion.get("explain", "")),
                str(suggestion.get("type", "")),
            ]
        )
        overlap_tokens = sorted(tokenize(text) & target_tokens)
        item = dict(suggestion)
        item["oracle_overlap"] = len(overlap_tokens)
        item["oracle_target_tokens"] = overlap_tokens
        scored.append(item)
    return scored


def rank_change_metrics(
    initial_rank: Optional[int],
    refined_rank: Optional[int],
) -> Dict[str, Any]:
    if initial_rank is None and refined_rank is None:
        status = "not_found"
        delta = None
    elif initial_rank is None and refined_rank is not None:
        status = "recovered_target"
        delta = None
    elif initial_rank is not None and refined_rank is None:
        status = "lost_target"
        delta = None
    elif refined_rank < initial_rank:
        status = "improved"
        delta = initial_rank - refined_rank
    elif refined_rank > initial_rank:
        status = "worsened"
        delta = initial_rank - refined_rank
    else:
        status = "unchanged"
        delta = 0

    return {
        "initial_rank": initial_rank,
        "refined_rank": refined_rank,
        "rank_delta": delta,
        "rank_change": status,
        "improved": int(status == "improved"),
        "worsened": int(status == "worsened"),
        "unchanged": int(status == "unchanged"),
        "lost_target": int(status == "lost_target"),
        "recovered_target": int(status == "recovered_target"),
    }


def combine_query(rewritten_query: str, suggestion: Optional[Dict[str, Any]]) -> str:
    if not suggestion:
        return rewritten_query

    sug = str(suggestion.get("sug", "")).strip()
    if not sug:
        return rewritten_query

    return f"{rewritten_query}; {sug}"


def best_scored_suggestion(scored_suggestions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not scored_suggestions:
        return None
    return max(
        scored_suggestions,
        key=lambda item: int(item.get("oracle_overlap", 0) or 0),
    )


def caption_only_evidence(image_ids, captions, scores, top_k: int) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "image_id": image_id,
            "score": score,
            "caption": caption,
            "visual_facts": [],
            "positive_facts": [],
            "negative_facts": [],
            "uncertain_facts": [],
            "enriched_caption": None,
        }
        for rank, (image_id, caption, score) in enumerate(
            zip(image_ids[:top_k], captions[:top_k], scores[:top_k]),
            start=1,
        )
    ]


def dedupe_keep_order(items: List[Any]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        clean = " ".join(str(item or "").strip().split())
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def evidence_facts(candidate: Dict[str, Any]) -> List[str]:
    return dedupe_keep_order(
        [
            *(candidate.get("visual_facts") or []),
            *(candidate.get("positive_facts") or []),
            *(candidate.get("negative_facts") or []),
            *(candidate.get("uncertain_facts") or []),
        ]
    )


def lexical_overlap_score(query_tokens: set[str], fact_tokens: set[str]) -> float:
    if not fact_tokens:
        return 0.0
    return len(query_tokens & fact_tokens) / len(fact_tokens)


def contradiction_penalty(query_tokens: set[str], fact_tokens: set[str]) -> float:
    penalty = 0.0
    for left, right in CONTRADICTION_PAIRS:
        if left in query_tokens and right in fact_tokens:
            penalty += 1.0
        if right in query_tokens and left in fact_tokens:
            penalty += 1.0

    negation_tokens = {"no", "not", "without"}
    if query_tokens & fact_tokens and fact_tokens & negation_tokens:
        penalty += 0.5
    return penalty


def discriminativeness_score(
    fact_tokens: set[str],
    candidate_token_df: Dict[str, int],
    num_candidates: int,
) -> float:
    if not fact_tokens or num_candidates <= 1:
        return 0.0

    scores = []
    for token in fact_tokens:
        df = candidate_token_df.get(token, 1)
        scores.append(1.0 - (df - 1) / max(1, num_candidates - 1))
    return sum(scores) / len(scores)


def clip_text_similarity_map(
    service: Any,
    query: str,
    facts: List[str],
) -> Dict[str, float]:
    if not facts:
        return {}

    embeddings = service.embed_texts([query, *facts])
    query_embedding = embeddings[0]
    fact_embeddings = embeddings[1:]
    similarities = (fact_embeddings @ query_embedding).tolist()
    return {
        fact: max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
        for fact, score in zip(facts, similarities)
    }


def select_query_aware_facts(
    service: Any,
    query: str,
    evidence: List[Dict[str, Any]],
    top_m: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> List[Dict[str, Any]]:
    """
    QAFS: Query-Aware Visual Fact Selection.
    Selects a compact, relevant, and discriminative subset of facts before
    sending candidate evidence to the RAIR reasoning LLM.
    """
    if top_m <= 0:
        return evidence

    all_facts = dedupe_keep_order(
        [
            fact
            for candidate in evidence
            for fact in evidence_facts(candidate)
        ]
    )
    clip_scores = clip_text_similarity_map(service, query, all_facts)
    query_tokens = tokenize(query)
    num_candidates = len(evidence)

    candidate_token_df: Dict[str, int] = {}
    for candidate in evidence:
        candidate_tokens = set()
        for fact in evidence_facts(candidate):
            candidate_tokens.update(tokenize(fact))
        for token in candidate_tokens:
            candidate_token_df[token] = candidate_token_df.get(token, 0) + 1

    selected_evidence = []
    for candidate in evidence:
        facts = evidence_facts(candidate)
        scored_facts = []
        for fact in facts:
            fact_tokens = tokenize(fact)
            components = {
                "clip": clip_scores.get(fact, 0.0),
                "lexical": lexical_overlap_score(query_tokens, fact_tokens),
                "discriminative": discriminativeness_score(
                    fact_tokens=fact_tokens,
                    candidate_token_df=candidate_token_df,
                    num_candidates=num_candidates,
                ),
                "contradiction": contradiction_penalty(query_tokens, fact_tokens),
            }
            score = (
                alpha * components["clip"]
                + beta * components["lexical"]
                + gamma * components["discriminative"]
                - delta * components["contradiction"]
            )
            scored_facts.append(
                {
                    "fact": fact,
                    "score": score,
                    "components": components,
                }
            )

        scored_facts.sort(key=lambda item: item["score"], reverse=True)
        selected_facts = [item["fact"] for item in scored_facts[:top_m]]
        selected_set = set(selected_facts)

        item = dict(candidate)
        item["visual_facts"] = selected_facts
        item["positive_facts"] = [
            fact for fact in candidate.get("positive_facts", []) if fact in selected_set
        ]
        item["negative_facts"] = [
            fact for fact in candidate.get("negative_facts", []) if fact in selected_set
        ]
        item["uncertain_facts"] = [
            fact for fact in candidate.get("uncertain_facts", []) if fact in selected_set
        ]
        item["enriched_caption"] = (
            f"{candidate.get('caption', '')}. Selected visual facts: "
            + "; ".join(selected_facts)
            if selected_facts
            else candidate.get("caption", "")
        )
        item["qafs"] = {
            "original_fact_count": len(facts),
            "selected_fact_count": len(selected_facts),
            "selected": scored_facts[:top_m],
            "weights": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
            },
        }
        selected_evidence.append(item)

    return selected_evidence


async def run_rewrite_only(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    rewritten_query: str,
    ks: List[int],
    search_depth: int,
) -> Dict[str, Any]:
    retrieved_ids, captions, scores = service.faiss_search(
        query_text=rewritten_query,
        gallery=gallery,
        top_k=search_depth,
    )
    rank = compute_rank(retrieved_ids, sample)
    return {
        "query": rewritten_query,
        **metrics_from_rank(rank, ks),
        "top_ids": retrieved_ids[: max(ks)],
        "top_scores": scores[: max(ks)],
        "top_captions": captions[: max(ks)],
    }


async def run_rair_variant(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    rewritten_query: str,
    ks: List[int],
    search_depth: int,
    evidence_top_k: int,
    with_facts: bool,
    use_qafs: bool,
    fact_top_m: int,
    fact_alpha: float,
    fact_beta: float,
    fact_gamma: float,
    fact_delta: float,
    oracle_overlap_threshold: int,
) -> Dict[str, Any]:
    image_ids, captions, scores = service.faiss_search(
        query_text=rewritten_query,
        gallery=gallery,
        top_k=search_depth,
    )
    initial_rank = compute_rank(image_ids, sample)

    if with_facts:
        evidence = service.build_candidate_evidence(
            image_ids=image_ids,
            captions=captions,
            scores=scores,
            top_k=evidence_top_k,
        )
    else:
        evidence = caption_only_evidence(
            image_ids=image_ids,
            captions=captions,
            scores=scores,
            top_k=evidence_top_k,
        )

    if with_facts and use_qafs:
        evidence = select_query_aware_facts(
            service=service,
            query=rewritten_query,
            evidence=evidence,
            top_m=fact_top_m,
            alpha=fact_alpha,
            beta=fact_beta,
            gamma=fact_gamma,
            delta=fact_delta,
        )

    reasoning = await service.reasoning(
        history=None,
        input=rewritten_query,
        candidate_evidence=evidence,
    )
    suggestions = reasoning.get("suggestions", [])
    scored_suggestions = score_oracle_suggestions(suggestions, sample)
    selected = choose_oracle_suggestion(
        scored_suggestions,
        sample,
        min_overlap=oracle_overlap_threshold,
    )
    rejected_best = None if selected else best_scored_suggestion(scored_suggestions)
    if rejected_best:
        rejected_best = dict(rejected_best)
        rejected_best["oracle_action"] = "reject"
        rejected_best["oracle_reason"] = (
            f"best suggestion overlap {rejected_best.get('oracle_overlap', 0)} "
            f"is below threshold {oracle_overlap_threshold}"
        )
    refined_query = combine_query(rewritten_query, selected)
    interaction_action = "accept" if selected else "no_op"

    refined_ids, refined_captions, refined_scores = service.faiss_search(
        query_text=refined_query,
        gallery=gallery,
        top_k=search_depth,
    )
    refined_rank = compute_rank(refined_ids, sample)

    return {
        "initial_query": rewritten_query,
        "refined_query": refined_query,
        "interaction_action": interaction_action,
        "oracle_overlap_threshold": oracle_overlap_threshold,
        "selected_suggestion": selected,
        "rejected_best_suggestion": rejected_best,
        "suggestions": scored_suggestions,
        "diagnosis": reasoning.get("diagnosis", {}),
        "candidate_evidence": evidence,
        "fact_selection": (
            {
                "method": "qafs",
                "top_m": fact_top_m,
                "alpha": fact_alpha,
                "beta": fact_beta,
                "gamma": fact_gamma,
                "delta": fact_delta,
            }
            if with_facts and use_qafs
            else {"method": "none"}
        ),
        **rank_change_metrics(initial_rank, refined_rank),
        **metrics_from_rank(refined_rank, ks),
        "top_ids": refined_ids[: max(ks)],
        "top_scores": refined_scores[: max(ks)],
        "top_captions": refined_captions[: max(ks)],
    }


async def evaluate_sample(
    service: Any,
    gallery: Dict[str, Any],
    sample: Dict[str, Any],
    methods: List[str],
    ks: List[int],
    search_depth: int,
    evidence_top_k: int,
    fact_top_m: int,
    fact_alpha: float,
    fact_beta: float,
    fact_gamma: float,
    fact_delta: float,
    oracle_overlap_threshold: int,
) -> Dict[str, Any]:
    rewritten_query, _ = await service.rewrite_query(build_rewrite_context(sample))

    method_results: Dict[str, Any] = {}
    if "rewrite_only" in methods:
        method_results["rewrite_only"] = await run_rewrite_only(
            service=service,
            gallery=gallery,
            sample=sample,
            rewritten_query=rewritten_query,
            ks=ks,
            search_depth=search_depth,
        )

    if "rair_without_facts" in methods:
        method_results["rair_without_facts"] = await run_rair_variant(
            service=service,
            gallery=gallery,
            sample=sample,
            rewritten_query=rewritten_query,
            ks=ks,
            search_depth=search_depth,
            evidence_top_k=evidence_top_k,
            with_facts=False,
            use_qafs=False,
            fact_top_m=fact_top_m,
            fact_alpha=fact_alpha,
            fact_beta=fact_beta,
            fact_gamma=fact_gamma,
            fact_delta=fact_delta,
            oracle_overlap_threshold=oracle_overlap_threshold,
        )

    if "rair_full" in methods:
        method_results["rair_full"] = await run_rair_variant(
            service=service,
            gallery=gallery,
            sample=sample,
            rewritten_query=rewritten_query,
            ks=ks,
            search_depth=search_depth,
            evidence_top_k=evidence_top_k,
            with_facts=True,
            use_qafs=False,
            fact_top_m=fact_top_m,
            fact_alpha=fact_alpha,
            fact_beta=fact_beta,
            fact_gamma=fact_gamma,
            fact_delta=fact_delta,
            oracle_overlap_threshold=oracle_overlap_threshold,
        )

    if "rair_full_qafs" in methods:
        method_results["rair_full_qafs"] = await run_rair_variant(
            service=service,
            gallery=gallery,
            sample=sample,
            rewritten_query=rewritten_query,
            ks=ks,
            search_depth=search_depth,
            evidence_top_k=evidence_top_k,
            with_facts=True,
            use_qafs=True,
            fact_top_m=fact_top_m,
            fact_alpha=fact_alpha,
            fact_beta=fact_beta,
            fact_gamma=fact_gamma,
            fact_delta=fact_delta,
            oracle_overlap_threshold=oracle_overlap_threshold,
        )

    return {
        "split": sample.get("split"),
        "dialog_index": sample.get("dialog_index"),
        "image_id": sample.get("image_id"),
        "image_path": sample.get("image_path"),
        "base_caption": sample.get("base_caption"),
        "rewritten_query": rewritten_query,
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
        logger.warning(
            "No target images from split=%s were found in the retrieval gallery. "
            "All Hit@K metrics will be 0.",
            args.split,
        )

    results = []
    for index, sample in enumerate(samples, start=1):
        logger.info(
            "E2 sample %d/%d split=%s dialog_index=%s image_id=%s",
            index,
            len(samples),
            args.split,
            sample.get("dialog_index"),
            sample.get("image_id"),
        )
        try:
            result = await evaluate_sample(
                service=service,
                gallery=gallery,
                sample=sample,
                methods=methods,
                ks=ks,
                search_depth=search_depth,
                evidence_top_k=args.evidence_top_k,
                fact_top_m=args.fact_top_m,
                fact_alpha=args.fact_alpha,
                fact_beta=args.fact_beta,
                fact_gamma=args.fact_gamma,
                fact_delta=args.fact_delta,
                oracle_overlap_threshold=args.oracle_overlap_threshold,
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
            "split": args.split,
            "limit": args.limit,
            "offset": args.offset,
            "methods": methods,
            "ks": ks,
            "search_depth": search_depth,
            "evidence_top_k": args.evidence_top_k,
            "fact_selection": {
                "qafs_top_m": args.fact_top_m,
                "alpha": args.fact_alpha,
                "beta": args.fact_beta,
                "gamma": args.fact_gamma,
                "delta": args.fact_delta,
            },
            "oracle_overlap_threshold": args.oracle_overlap_threshold,
            "clip_model": args.clip_model,
            "device": args.device,
            "llm_provider": args.llm_provider,
            "reasoning_model": args.reasoning_model,
            "local_llm_model": args.local_llm_model,
            "gallery_overlap": overlap,
            "interaction_policy": "target_fact_overlap_accept_or_no_op",
        },
        "summary": summarize_e2(results, methods, ks),
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2 main RAIR comparison: Rewrite-only vs RAIR without facts vs RAIR full."
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to evaluate. Keep small because this calls an LLM.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--evidence-top-k", type=int, default=8)
    parser.add_argument(
        "--oracle-overlap-threshold",
        type=int,
        default=1,
        help="Minimum target-fact token overlap required to accept a generated suggestion.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rewrite_only", "rair_without_facts", "rair_full", "rair_full_qafs"],
        default=["rewrite_only", "rair_without_facts", "rair_full"],
    )
    parser.add_argument("--fact-top-m", type=int, default=4)
    parser.add_argument("--fact-alpha", type=float, default=0.5)
    parser.add_argument("--fact-beta", type=float, default=0.3)
    parser.add_argument("--fact-gamma", type=float, default=0.2)
    parser.add_argument("--fact-delta", type=float, default=0.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=resolve_torch_device("CLIP_DEVICE"))
    parser.add_argument("--llm-provider", choices=["openai", "local"], default=os.environ.get("RAIR_LLM_PROVIDER", "openai"))
    parser.add_argument("--reasoning-model", default=os.environ.get("RAIR_REASONING_MODEL", "gpt-5.4"))
    parser.add_argument("--local-llm-model", default=os.environ.get("RAIR_LOCAL_LLM_MODEL", "google/gemma-3-4b-it"))
    parser.add_argument("--local-llm-device", default=resolve_torch_device("LOCAL_LLM_DEVICE"))
    parser.add_argument("--local-llm-dtype", default=os.environ.get("LOCAL_LLM_DTYPE", "auto"))
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--local-llm-online",
        action="store_true",
        help="Allow HuggingFace downloads/lookups for the local LLM if it is not cached.",
    )
    parser.add_argument("--output", type=Path, default=Path("e2_rair_comparison_results.json"))
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
