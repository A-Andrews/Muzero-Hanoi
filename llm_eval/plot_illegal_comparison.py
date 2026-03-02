#!/usr/bin/env python3
"""Illegal move rate comparison: MuZero ablations vs LLMs.

Parallels Grafman et al. (1992) finding that cerebellar patients made
significantly more illegal moves on the Tower of Hanoi.

Produces a grouped bar chart comparing illegal move rates across
MuZero conditions and LLM prompting strategies per difficulty.

Usage:
    python llm_eval/plot_illegal_comparison.py \
        --timestamp 1748875208 \
        --model_label qwen25_7b \
        --no_latex
"""

import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import PLOT_COLORS, set_plot_style

# ── Constants ────────────────────────────────────────────────────────────

DIFFICULTIES = [
    ("LS", "Close"),
    ("MS", "Moderate"),
    ("ES", "Far"),
]

# MuZero illegal move rates from existing evaluation
# (illegal_move_rate_comparison.py with n_sims=25, random starts, 100 episodes)
# Source: stats/Hanoi/1748875208/illegal_move_rate_1748875208.png
# These are mean ± SE (SE computed as std(ddof=1)/sqrt(n))
MUZERO_ILLEGAL_RATES = {
    "aggregate": {
        "MuZero": (16.5, 1.5),          # ~16.5% ± 1.5%
        "Policy ablated\n(Cerebellar)": (47.0, 1.5),  # ~47% ± 1.5%
        "Value ablated\n(PFC lesion)": (20.5, 1.5),   # ~20.5% ± 1.5%
    }
}


def load_llm_json(directory: str, file_stem: str) -> dict | None:
    path = os.path.join(directory, file_stem + "_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_per_difficulty_muzero(root_dir: str) -> dict | None:
    """Load per-difficulty MuZero illegal rates if available (from extended runs)."""
    results = {}
    for diff_dir, _ in DIFFICULTIES:
        path = os.path.join(root_dir, diff_dir, "muzero_illegal_rates.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results[diff_dir] = json.load(f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot illegal move rate comparison")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model_labels", nargs="+", required=True)
    parser.add_argument("--llm_display_names", nargs="+", default=None)
    parser.add_argument("--prompting_strategies", nargs="+",
                        default=["zero_shot", "cot"])
    parser.add_argument("--no_latex", action="store_true")
    args = parser.parse_args()

    set_plot_style()
    if args.no_latex:
        mpl.rcParams["text.usetex"] = False

    font_s = 7
    mpl.rc("font", size=font_s)
    mpl.rcParams["xtick.labelsize"] = font_s
    mpl.rcParams["ytick.labelsize"] = font_s

    root_dir = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp)

    n_llm = len(args.model_labels) * len(args.prompting_strategies)
    if args.llm_display_names is not None:
        display_names = list(args.llm_display_names)
    else:
        display_names = []
    while len(display_names) < n_llm:
        idx = len(display_names)
        m_idx = idx // len(args.prompting_strategies)
        s_idx = idx % len(args.prompting_strategies)
        display_names.append(
            f"{args.model_labels[m_idx]} ({args.prompting_strategies[s_idx]})"
        )

    extra_colors = ["#8EC8C8", "#D4A5D4", "#F5C281", "#A8D5A2"]


    # Check if per-difficulty MuZero data exists
    per_diff_muzero = load_per_difficulty_muzero(root_dir)

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=True)

    muzero_conditions = [
        ("MuZero", PLOT_COLORS[0]),
        ("Value ablated\n(PFC lesion)", PLOT_COLORS[2]),
        ("Policy ablated\n(Cerebellar)", PLOT_COLORS[1]),
    ]

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        # MuZero conditions
        for cond_name, cond_color in muzero_conditions:
            if per_diff_muzero is not None:
                d = per_diff_muzero[diff_dir]
                rate = d[cond_name]["mean"] * 100
                se = d[cond_name]["se"] * 100
            else:
                # Use aggregate values
                rate, se = MUZERO_ILLEGAL_RATES["aggregate"][cond_name]
            names.append(cond_name)
            rates.append(rate)
            ses.append(se)
            colors.append(cond_color)

        # LLM conditions (iterate models × strategies)
        llm_idx = 0
        for model_label in args.model_labels:
            for strat in args.prompting_strategies:
                file_stem = f"LLM_{model_label}_{strat}"
                disp_name = display_names[llm_idx]
                col_c = extra_colors[llm_idx % len(extra_colors)]
                llm_idx += 1
                data = load_llm_json(file_dir, file_stem)
                if data is None:
                    continue
                names.append(disp_name)
                rates.append(data["mean_illegal_rate"] * 100)
                ses.append(data["se_illegal_rate"] * 100)
                colors.append(col_c)

        ax = axs[col]
        x = np.arange(len(names))
        width = 0.6
        bars = ax.bar(
            x, rates, width, yerr=ses, capsize=4,
            color=colors, edgecolor="none",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=font_s)
        ax.set_title(diff_title)
        if col == 0:
            ax.set_ylabel("Illegal move rate (%)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Auto-scale y with headroom
        max_bar = max(r + s for r, s in zip(rates, ses)) if rates else 1.0
        ax.set_ylim(0, max_bar * 1.2)

        # Value labels
        for bar, val in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_bar * 0.02,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=font_s - 1,
            )

    note = "(MuZero: aggregate over random starts, n_sims=25)"
    if per_diff_muzero is not None:
        note = "(MuZero: per-difficulty, n_sims=150)"
    fig.text(0.5, -0.02, note, ha="center", fontsize=font_s - 1, style="italic")

    fig.tight_layout()
    out_path = os.path.join(root_dir, f"IllegalRate_Comparison_{args.timestamp}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
