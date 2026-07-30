import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E3_SCRIPT = PROJECT_ROOT / "Experiments" / "e3_turn_by_turn.py"


QVFS_VARIANTS: Dict[str, Dict[str, object]] = {
    "no_qvfs": {
        "label": "No QVFS",
        "method": "rair_full",
        "weights": None,
    },
    "wo_semantic": {
        "label": "w/o semantic",
        "method": "rair_full_qvfs",
        "weights": {"alpha": 0.0, "beta": 0.3, "gamma": 0.2, "delta": 0.5},
    },
    "wo_lexical": {
        "label": "w/o lexical",
        "method": "rair_full_qvfs",
        "weights": {"alpha": 0.5, "beta": 0.0, "gamma": 0.2, "delta": 0.5},
    },
    "wo_discriminative": {
        "label": "w/o discriminative",
        "method": "rair_full_qvfs",
        "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.0, "delta": 0.5},
    },
    "wo_contradiction": {
        "label": "w/o contradiction",
        "method": "rair_full_qvfs",
        "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "delta": 0.0},
    },
    "full_qvfs": {
        "label": "Full QVFS",
        "method": "rair_full_qvfs",
        "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "delta": 0.5},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run QVFS ablation studies by launching E3 once per variant. "
            "Defaults are set for SigLIP fusion lambda=0.9."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(QVFS_VARIANTS),
        default=list(QVFS_VARIANTS),
        help="QVFS variants to run, in order.",
    )
    parser.add_argument("--dataset", choices=["visdial", "flickr30k"], default="flickr30k")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--embedding-backend", choices=["clip", "siglip"], default="siglip")
    parser.add_argument("--retrieval-index", choices=["image", "caption", "fusion"], default="fusion")
    parser.add_argument("--fusion-alpha", type=float, default=0.9)
    parser.add_argument("--fusion-pool-size", type=int, default=200)
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--fact-top-m", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--stop-on-hit-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "local"], default="local")
    parser.add_argument("--local-llm-model", default="google/gemma-3-12b-it")
    parser.add_argument("--local-llm-device", default="cuda")
    parser.add_argument("--local-llm-dtype", default="bfloat16")
    parser.add_argument("--user-sim-provider", choices=["text", "vlm"], default="vlm")
    parser.add_argument("--user-sim-vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--user-sim-vlm-device", default="cuda")
    parser.add_argument("--user-sim-vlm-dtype", default="bfloat16")
    parser.add_argument("--target-observation-provider", choices=["none", "openai", "local"], default="none")
    parser.add_argument("--target-observation-image-root", default=None)
    parser.add_argument("--initial-query-source", choices=["auto", "base_caption", "vlm_user"], default="auto")
    parser.add_argument("--selection-policy", choices=["oracle_overlap", "llm_user_sim"], default="llm_user_sim")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qvfs_ablation")
    parser.add_argument("--log-dir", type=Path, default=Path("logs") / "qvfs_ablation")
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--save-evidence", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_name_prefix(args: argparse.Namespace) -> str:
    alpha = str(args.fusion_alpha).replace(".", "")
    return (
        f"e3_{args.dataset}_{args.embedding_backend}_{args.split}_{args.limit}"
        f"_t{args.turns}_{args.retrieval_index}_a{alpha}_qvfs_ablation"
    )


def build_command(args: argparse.Namespace, variant_name: str) -> List[str]:
    variant = QVFS_VARIANTS[variant_name]
    name_prefix = args.name_prefix or default_name_prefix(args)
    output = args.output_dir / f"{name_prefix}_{variant_name}.json"
    conversation_log_dir = args.log_dir / f"{name_prefix}_{variant_name}"

    command = [
        args.python,
        str(E3_SCRIPT),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
        "--turns",
        str(args.turns),
        "--initial-query-source",
        args.initial_query_source,
        "--search-depth",
        str(args.search_depth),
        "--retrieval-index",
        args.retrieval_index,
        "--fusion-alpha",
        str(args.fusion_alpha),
        "--fusion-pool-size",
        str(args.fusion_pool_size),
        "--evidence-top-k",
        str(args.evidence_top_k),
        "--methods",
        str(variant["method"]),
        "--fact-top-m",
        str(args.fact_top_m),
        "--embedding-backend",
        args.embedding_backend,
        "--device",
        args.device,
        "--llm-provider",
        args.llm_provider,
        "--local-llm-model",
        args.local_llm_model,
        "--local-llm-device",
        args.local_llm_device,
        "--local-llm-dtype",
        args.local_llm_dtype,
        "--selection-policy",
        args.selection_policy,
        "--user-sim-provider",
        args.user_sim_provider,
        "--user-sim-vlm-model",
        args.user_sim_vlm_model,
        "--user-sim-vlm-device",
        args.user_sim_vlm_device,
        "--user-sim-vlm-dtype",
        args.user_sim_vlm_dtype,
        "--target-observation-provider",
        args.target_observation_provider,
        "--stop-on-hit-k",
        str(args.stop_on_hit_k),
        "--conversation-log-dir",
        str(conversation_log_dir),
        "--output",
        str(output),
        "--continue-on-error",
    ]

    if args.target_observation_image_root:
        command.extend(["--target-observation-image-root", args.target_observation_image_root])
    if args.save_evidence:
        command.append("--save-evidence")

    weights: Optional[Dict[str, float]] = variant["weights"]  # type: ignore[assignment]
    if weights:
        command.extend(
            [
                "--fact-alpha",
                str(weights["alpha"]),
                "--fact-beta",
                str(weights["beta"]),
                "--fact-gamma",
                str(weights["gamma"]),
                "--fact-delta",
                str(weights["delta"]),
            ]
        )

    return command


def summarize_result(path: Path, variant_name: str) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    method_payload = next(iter(payload["summary"]["methods"].values()))
    final = method_payload["final"]
    return {
        "variant": variant_name,
        "label": QVFS_VARIANTS[variant_name]["label"],
        "mrr": final.get("mrr"),
        "hit@1": final.get("hit@1"),
        "hit@5": final.get("hit@5"),
        "hit@10": final.get("hit@10"),
        "hit@20": final.get("hit@20"),
        "hit@50": final.get("hit@50"),
        "found_rate": final.get("found_rate"),
        "output": str(path),
    }


def command_output_path(command: List[str]) -> Path:
    return Path(command[command.index("--output") + 1])


def print_command(command: List[str]) -> None:
    print(" ".join(f'"{item}"' if " " in item else item for item in command))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    summaries = []
    for variant_name in args.variants:
        command = build_command(args, variant_name)
        output_path = command_output_path(command)

        print(f"\n=== Running {variant_name}: {QVFS_VARIANTS[variant_name]['label']} ===")
        print_command(command)

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
        if completed.returncode != 0:
            message = f"Variant failed: {variant_name} exit_code={completed.returncode}"
            if args.continue_on_error:
                print(message, file=sys.stderr)
                continue
            raise SystemExit(message)

        if output_path.exists():
            summaries.append(summarize_result(output_path, variant_name))

    if args.dry_run:
        return

    summary_output = args.summary_output or (args.output_dir / f"{args.name_prefix or default_name_prefix(args)}_summary.json")
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump({"variants": summaries}, f, ensure_ascii=False, indent=2)

    print(f"\nAblation summary written to: {summary_output}")
    print(json.dumps({"variants": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
