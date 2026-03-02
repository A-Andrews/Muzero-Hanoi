"""Plot layerwise LLM intervention results: error vs. layer index.

Loads results from the layer sweep (run_llm_layer_sweep.sh) and plots:
  - Ablation curve: mean_error vs. layer index (with SE error bars)
  - Noise curve: mean_error vs. layer index (with SE error bars)
  - Baseline (no intervention) as a horizontal dashed line

Output: stats/Hanoi/<timestamp>/<difficulty>/LLM_LayerSweep_<timestamp>.png

Usage:
    python llm_eval/plot_layer_ablation.py \
        --timestamp <timestamp> \
        --model_label llama3_8b \
        --start 0 \
        --noise_scale 0.5
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils import PLOT_COLORS, set_plot_style

parser = argparse.ArgumentParser(description="Plot layerwise LLM ablation results")
parser.add_argument("--timestamp", type=str, required=True)
parser.add_argument("--model_label", type=str, required=True, help="e.g. llama3_8b")
parser.add_argument("--start", type=int, default=0, choices=[0, 1, 2],
                    help="0=Far(ES), 1=Moderate(MS), 2=Close(LS)")
parser.add_argument("--noise_scale", type=float, default=0.5,
                    help="Noise scale used in the noise sweep (for filename lookup)")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31],
                    help="Layer indices that were swept (must match run_llm_layer_sweep.sh)")
args = parser.parse_args()

DIFFICULTY_MAP = {0: "ES", 1: "MS", 2: "LS"}
DIFFICULTY_NAME = {0: "Far (7 moves)", 1: "Moderate (3 moves)", 2: "Close (1 move)"}

diff_dir = DIFFICULTY_MAP[args.start]
root_dir = os.path.join("stats", "Hanoi", args.timestamp, diff_dir)


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_result(label: str, prompting: str) -> tuple[float, float] | None:
    """Return (mean_error, se_error) from JSON, or None if file missing."""
    fname = f"LLM_{label}_{prompting}_results.json"
    data = load_json(os.path.join(root_dir, fname))
    if data is None:
        return None
    return data["mean_error"], data["se_error"]


# --- Baseline (no intervention) ---
baseline = load_result(args.model_label, "cot")

# --- Ablation sweep ---
ablation_means, ablation_ses, ablation_layers = [], [], []
for layer in args.layers:
    prompting = f"cot_ablateL{layer}"
    res = load_result(args.model_label, prompting)
    if res is not None:
        ablation_means.append(res[0])
        ablation_ses.append(res[1])
        ablation_layers.append(layer)
    else:
        print(f"Warning: missing ablation result for layer {layer}")

# --- Noise sweep ---
noise_means, noise_ses, noise_layers = [], [], []
for layer in args.layers:
    nl_str = str(layer)
    prompting = f"cot_noiseS{args.noise_scale}_L{nl_str}"
    res = load_result(args.model_label, prompting)
    if res is not None:
        noise_means.append(res[0])
        noise_ses.append(res[1])
        noise_layers.append(layer)
    else:
        print(f"Warning: missing noise result for layer {layer}")

if not ablation_layers and not noise_layers:
    print("No layer sweep results found. Run run_llm_layer_sweep.sh first.")
    sys.exit(1)

set_plot_style()

font_s = 8
mpl.rc("font", size=font_s)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

color_ablation = PLOT_COLORS[2]   # red-ish
color_noise = PLOT_COLORS[1]      # blue-ish
color_baseline = PLOT_COLORS[0]   # black

# Baseline horizontal line
if baseline is not None:
    ax.axhline(
        baseline[0],
        color=color_baseline,
        linestyle="--",
        linewidth=1.5,
        label=f"No intervention ({baseline[0]:.1f})",
        zorder=1,
    )
    ax.fill_between(
        [min(args.layers) - 1, max(args.layers) + 1],
        baseline[0] - baseline[1],
        baseline[0] + baseline[1],
        color=color_baseline,
        alpha=0.1,
        zorder=0,
    )

if ablation_layers:
    ax.errorbar(
        ablation_layers,
        ablation_means,
        yerr=ablation_ses,
        fmt="-o",
        color=color_ablation,
        linewidth=2.0,
        markersize=5,
        capsize=3,
        label="Layer ablation (skip)",
        zorder=3,
    )

if noise_layers:
    ax.errorbar(
        noise_layers,
        noise_means,
        yerr=noise_ses,
        fmt="-s",
        color=color_noise,
        linewidth=2.0,
        markersize=5,
        capsize=3,
        label=f"Noise injection ($\\sigma={args.noise_scale}$)",
        zorder=2,
    )

ax.set_xlabel("Transformer layer index", fontsize=font_s + 1)
ax.set_ylabel("Mean error (steps above optimal)", fontsize=font_s + 1)
ax.set_title(
    f"LLM layerwise intervention — {DIFFICULTY_NAME[args.start]} (CoT)",
    fontsize=font_s + 2,
)
ax.set_xticks(args.layers)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=font_s, frameon=False)

fig.tight_layout()
out_path = os.path.join(root_dir, f"LLM_LayerSweep_{args.timestamp}.png")
fig.savefig(out_path, dpi=1200, bbox_inches="tight")
print(f"Saved: {out_path}")
