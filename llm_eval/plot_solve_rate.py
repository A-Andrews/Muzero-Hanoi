#!/usr/bin/env python3
"""Solve rate comparison: MuZero ablations vs LLMs across difficulties.

Produces a 3-panel bar chart (Close / Moderate / Far) showing the
percentage of episodes solved by each agent condition.  Directly
mirrors the primary outcome metric in the human lesion studies
(Goel & Grafman 1995; Grafman et al. 1992).

Usage:
    python llm_eval/plot_solve_rate.py \
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
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import PLOT_COLORS, set_plot_style

# ── Constants ────────────────────────────────────────────────────────────

MUZERO_CONDITIONS = [
    ("Muzero", "MuZero"),
    ("ResetLatentVal", "Value ablated\n(PFC lesion)"),
    ("ResetLatentPol", "Policy ablated\n(Cerebellar)"),
]

DIFFICULTIES = [
    ("LS", "Close"),
    ("MS", "Moderate"),
    ("ES", "Far"),
]

# max_steps=200; optimal moves per difficulty
OPTIMAL_MOVES = {"LS": 1, "MS": 3, "ES": 7}
MAX_STEPS = 200

# ── Data loading ─────────────────────────────────────────────────────────


def load_accuracy(directory: str, label: str) -> np.ndarray | None:
    """Load *_actingAccuracy(_error).pt → array with cols [n_sims, mean(, std)]."""
    path = os.path.join(directory, label + "_actingAccuracy.pt")
    if not os.path.exists(path):
        return None
    arr = torch.load(path, weights_only=False)
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=float)

    err_path = os.path.join(directory, label + "_actingAccuracy_error.pt")
    if os.path.exists(err_path):
        err = torch.load(err_path, weights_only=False)
        if torch.is_tensor(err):
            err = err.cpu().numpy()
        err = np.asarray(err, dtype=float)
        if err.shape[1] >= 3:
            arr = np.column_stack([arr, err[:, 2]])
    return arr


def load_llm_json(directory: str, file_stem: str) -> dict | None:
    path = os.path.join(directory, file_stem + "_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def muzero_solve_rate(
    arr: np.ndarray,
    diff_key: str,
    muzero_runs: int,
) -> tuple[float, float]:
    """Estimate solve rate from MuZero error data at n_sims=150 (last row).

    An episode that hits max_steps=200 yields error = 200 - optimal.
    If the mean error is well below this threshold, solve rate ≈ 1.0.
    """
    failed_error = MAX_STEPS - OPTIMAL_MOVES[diff_key]
    mean_err = arr[-1, 1]
    sd_err = arr[-1, 2] if arr.shape[1] > 2 else 0.0

    # Heuristic: if mean + 2*SD is well below failed_error, all runs solved.
    # If mean - 2*SD is near failed_error, most runs failed.
    # For intermediate cases, estimate fraction of runs below threshold.
    if mean_err + 2 * sd_err < failed_error * 0.8:
        return 1.0, 0.0
    if mean_err - 2 * sd_err > failed_error * 0.8:
        return 0.0, 0.0

    # Rough estimate: assume errors are normally distributed across runs
    # and count what fraction falls below the failed threshold.
    # With only 5 runs this is approximate.
    from scipy.stats import norm

    if sd_err > 0:
        solve_prob = norm.cdf(failed_error - 1, loc=mean_err, scale=sd_err)
    else:
        solve_prob = 1.0 if mean_err < failed_error - 1 else 0.0

    # SE of a proportion with n=muzero_runs
    se = np.sqrt(solve_prob * (1 - solve_prob) / muzero_runs) if muzero_runs > 1 else 0.0
    return float(solve_prob), float(se)


# ── Plotting ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot solve rate comparison")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model_labels", nargs="+", required=True)
    parser.add_argument("--llm_display_names", nargs="+", default=None)
    parser.add_argument("--prompting_strategies", nargs="+",
                        default=["zero_shot", "cot"])
    parser.add_argument("--no_latex", action="store_true")
    parser.add_argument("--muzero_runs", type=int, default=5)
    args = parser.parse_args()

    set_plot_style()
    if args.no_latex:
        mpl.rcParams["text.usetex"] = False

    font_s = 7
    mpl.rc("font", size=font_s)
    mpl.rcParams["xtick.labelsize"] = font_s
    mpl.rcParams["ytick.labelsize"] = font_s

    root_dir = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp)

    extra_colors = ["#8EC8C8", "#D4A5D4", "#F5C281", "#A8D5A2"]
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

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=True)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        # MuZero conditions
        for i, (label, name) in enumerate(MUZERO_CONDITIONS):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            sr, se = muzero_solve_rate(arr, diff_dir, args.muzero_runs)
            names.append(name)
            rates.append(sr * 100)
            ses.append(se * 100)
            colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

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
                sr = data["solve_rate"]
                se_sr = data.get("se_solve_rate", 0.0)
                names.append(disp_name)
                rates.append(sr * 100)
                ses.append(se_sr * 100)
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
            ax.set_ylabel("Solve rate (%)")
        ax.set_ylim(0, 115)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Value labels above bars
        for bar, val in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=font_s - 1,
            )

    fig.tight_layout()
    out_path = os.path.join(root_dir, f"SolveRate_Comparison_{args.timestamp}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
