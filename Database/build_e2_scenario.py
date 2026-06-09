import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Database.db_session import SessionLocal
from Entities.entities import VisDialTargetAnnotations


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "rair" / "scenarios"


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [
            str(item).strip()
            for item in value.values()
            if str(item).strip()
        ]
    text = str(value).strip()
    return [text] if text else []


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
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


def target_facts_for_sample(row: VisDialTargetAnnotations) -> List[str]:
    return dedupe_keep_order(
        [
            row.base_caption or "",
            *ensure_list(row.visual_facts),
            *ensure_list(row.positive_facts),
            *ensure_list(row.negative_facts),
            *ensure_list(row.uncertain_facts),
        ]
    )


def row_to_scenario_sample(row: VisDialTargetAnnotations, sample_index: int) -> Dict[str, Any]:
    target_facts = target_facts_for_sample(row)
    return {
        "sample_index": sample_index,
        "sample_id": f"{row.split}:{row.dialog_index}:{row.image_id}",
        "split": row.split,
        "dialog_index": row.dialog_index,
        "image_id": str(row.image_id) if row.image_id is not None else None,
        "image_path": row.image_path,
        "initial_query": row.base_caption or "",
        "base_caption": row.base_caption or "",
        "enriched_caption": row.enriched_caption or "",
        "target_facts": target_facts,
        "positive_facts": ensure_list(row.positive_facts),
        "negative_facts": ensure_list(row.negative_facts),
        "uncertain_facts": ensure_list(row.uncertain_facts),
        "oracle": {
            "visible_to": "simulated_user_only",
            "accept_rule": "accept the best generated suggestion only when it is supported by target_facts",
            "reject_rule": "reject all suggestions and keep the query unchanged when no suggestion reaches min_overlap",
        },
    }


def load_rows(
    split: str,
    limit: Optional[int],
    offset: int,
    shuffle: bool,
    seed: int,
) -> List[VisDialTargetAnnotations]:
    with SessionLocal() as session:
        stmt = (
            select(VisDialTargetAnnotations)
            .where(VisDialTargetAnnotations.split == split)
            .order_by(VisDialTargetAnnotations.dialog_index)
            .offset(offset)
        )
        if limit is not None and limit >= 0 and not shuffle:
            stmt = stmt.limit(limit)

        rows = list(session.execute(stmt).scalars().all())

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
        if limit is not None and limit >= 0:
            rows = rows[:limit]

    return rows


def build_scenario(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not rows:
        raise ValueError(f"No VisDialTargetAnnotations rows found for split={args.split}")

    samples = [
        row_to_scenario_sample(row, sample_index=index)
        for index, row in enumerate(rows)
    ]

    return {
        "scenario_name": args.scenario_name,
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_table": "VisDialTargetAnnotations",
        "description": (
            "Target-aware simulated interaction scenario for E2 RAIR. "
            "The RAIR system must not see target_facts; they are reserved for "
            "the simulated user/oracle that accepts or rejects generated suggestions."
        ),
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "shuffle": args.shuffle,
        "seed": args.seed if args.shuffle else None,
        "policy": {
            "experiment": "E2",
            "max_turns": 1,
            "oracle_type": "target_fact_overlap_accept_or_no_op",
            "min_overlap": args.min_overlap,
            "allow_no_op": True,
            "candidate_evidence_top_k": args.evidence_top_k,
            "target_fact_visibility": "simulated_user_only",
            "system_visibility": [
                "initial_query",
                "retrieved_candidate_captions",
                "retrieved_candidate_facts_when_enabled",
            ],
        },
        "samples": samples,
    }


def default_output_path(args: argparse.Namespace) -> Path:
    limit_name = "full" if args.limit is None or args.limit < 0 else str(args.limit)
    suffix = "shuffled" if args.shuffle else "ordered"
    filename = (
        f"e2_{args.split}_{limit_name}_top{args.evidence_top_k}_"
        f"oracle{args.min_overlap}_{suffix}.json"
    )
    return DEFAULT_OUTPUT_DIR / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen E2 target-aware simulated interaction scenario JSON."
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence-top-k", type=int, default=3)
    parser.add_argument("--min-overlap", type=int, default=1)
    parser.add_argument(
        "--scenario-name",
        default="rair_e2_target_fact_oracle_v1",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        args.limit = None

    scenario = build_scenario(args)
    output_path = args.output or default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "scenario_name": scenario["scenario_name"],
                "split": scenario["split"],
                "num_samples": len(scenario["samples"]),
                "policy": scenario["policy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
