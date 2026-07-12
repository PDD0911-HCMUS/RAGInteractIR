import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

data = """Turn\tCLIP λ=0.8 Hit@10\tCLIP λ=0.8 Hit@20\tCLIP λ=0.8 Hit@50\tCLIP λ=0.9 Hit@10\tCLIP λ=0.9 Hit@20\tCLIP λ=0.9 Hit@50\tSigLIP λ=0.9 Hit@10\tSigLIP λ=0.9 Hit@20\tSigLIP λ=0.9 Hit@50
0\t0.541\t0.6312\t0.7399\t0.5481\t0.6336\t0.7435\t0.7607\t0.8183\t0.889
1\t0.5736\t0.6645\t0.766\t0.576\t0.6621\t0.7738\t0.7797\t0.8373\t0.905
2\t0.5814\t0.6692\t0.7708\t0.5855\t0.6704\t0.7767\t0.7838\t0.8414\t0.9091
3\t0.5831\t0.671\t0.7743\t0.5891\t0.6734\t0.7803\t0.7862\t0.8426\t0.9103
4\t0.5855\t0.6722\t0.7767\t0.5909\t0.6752\t0.7821\t0.7868\t0.8438\t0.9103
5\t0.5855\t0.6722\t0.7773\t0.5914\t0.6758\t0.7821\t0.788\t0.8462\t0.9115
"""

df = pd.read_csv(StringIO(data), sep="\t")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)

# Phân biệt metric bằng linestyle + marker
style_map = {
    "Hit@10": {"linestyle": "-",  "marker": "o"},
    "Hit@20": {"linestyle": "--", "marker": "s"},
    "Hit@50": {"linestyle": ":",  "marker": "^"},
}

# Phân biệt backbone + lambda bằng màu
color_map = {
    ("CLIP", "0.8"): "#7FB3FF",    # xanh nhạt
    ("CLIP", "0.9"): "#1F5FBF",    # xanh đậm
    ("SigLIP", "0.8"): "#FFBE7A",  # cam nhạt (để dành khi có dữ liệu)
    ("SigLIP", "0.9"): "#D97706",  # cam đậm
}

for col in df.columns:
    if col == "Turn":
        continue

    parts = col.split()
    backbone = parts[0]          # CLIP / SigLIP
    lam = parts[1].split("=")[1] # 0.8 / 0.9
    metric = parts[2]            # Hit@10 / Hit@20 / Hit@50

    style = style_map[metric]
    color = color_map[(backbone, lam)]

    ax.plot(
        df["Turn"],
        df[col],
        label=col,
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=2.3,
        markersize=5.5,
    )

ax.set_title("Best-so-far Hit@K across Interaction Turns")
ax.set_xlabel("Interaction Turn")
ax.set_ylabel("Hit@K")
ax.set_xticks(df["Turn"])

# Nếu muốn dễ nhìn hơn
ax.set_ylim(0.5, 0.95)

# Nếu muốn trục chuẩn tuyệt đối thì dùng:
# ax.set_ylim(0.0, 1.0)

ax.grid(axis="y", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
)

fig.tight_layout()

out_dir = Path(r"F:\RAGInteractIR\docs")
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "multiturn_hitk_all_lines.png", bbox_inches="tight", dpi=300)
fig.savefig(out_dir / "multiturn_hitk_all_lines.svg", bbox_inches="tight")

plt.show()