import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Experiments.e3_turn_by_turn import summarize_e3


METHOD = "rair_full_qvfs"
DEFAULT_METRICS = [
    ("mrr", "MRR"),
    ("hit@10", "Hit@10"),
    ("hit@20", "Hit@20"),
    ("hit@50", "Hit@50"),
]


def load_turn_series(path: Path, metrics: List[Tuple[str, str]]) -> Dict[str, List[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if data.get("results"):
        config = data.get("config") or {}
        turns_arg = int(config.get("turns") or 5)
        ks = config.get("ks") or [1, 5, 10, 20, 50]
        methods = [METHOD]
        summary = summarize_e3(data["results"], methods, ks, turns_arg)

    turns = summary["methods"][METHOD]["turns"]
    max_turn = max(int(key) for key in turns.keys())

    series = {metric_key: [] for metric_key, _ in metrics}
    for turn_id in range(max_turn + 1):
        item = turns[str(turn_id)]
        for metric_key, _ in metrics:
            series[metric_key].append(float(item.get(metric_key, 0.0)))
    return series


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_chart(
    input_files: List[Path],
    labels: List[str],
    output_prefix: Path,
) -> None:
    configure_matplotlib()

    metrics = DEFAULT_METRICS
    loaded = [load_turn_series(path, metrics) for path in input_files]
    max_turn = max(len(item[metrics[0][0]]) for item in loaded) - 1
    turns = list(range(max_turn + 1))

    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metrics):
        for idx, (label, series) in enumerate(zip(labels, loaded)):
            values = series[metric_key]
            padded = values + [values[-1]] * (len(turns) - len(values))
            ax.plot(
                turns,
                padded[: len(turns)],
                label=label,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=1.8,
                markersize=4.2,
            )
        ax.set_title(title, pad=4)
        ax.set_xlabel("Interaction turn")
        ax.set_ylabel(title)
        ax.set_xticks(turns)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=min(3, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        output_path = output_prefix.with_suffix(f".{ext}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E3 retrieval backend comparison line charts."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="E3 JSON result files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels matching --inputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("docs/e3_retrieval_backend_line_chart"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")
    plot_chart(args.inputs, args.labels, args.output_prefix)


if __name__ == "__main__":
    main()
