#!/usr/bin/env python3
"""Overlay LLM results on the MuZero ablation bar charts.

Produces two figures that can go directly into the paper:

Figure 1 — Mean excess moves (error) × difficulty × agent
  Three panels (Close / Moderate / Far), bars for each agent.

Figure 2 — Illegal-move rate × difficulty × agent
  Same layout as Figure 1.

Both figures follow the same visual style as the existing MuZero plots
(set_plot_style, PLOT_COLORS).

Usage:
    python llm_eval/plot_llm_comparison.py \\
        --timestamp 1748875208 \\
        --model_label llama3_8b \\
        --llm_names "Llama-3 8B (zero-shot)" "Llama-3 8B (CoT)"

The script automatically discovers which LLM conditions are present under
stats/Hanoi/<timestamp>/<difficulty>/ and plots them alongside the MuZero
baselines.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import PLOT_COLORS, set_plot_style

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

# MuZero conditions in the order used by the existing plotting code
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


def load_accuracy(directory: str, label: str) -> np.ndarray | None:
    """Load *_actingAccuracy(_error).pt and return array with cols [n_sims, mean(, std)].

    Returns None if the file does not exist.
    """
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
    """Load full JSON results for an LLM condition.  Returns None if absent."""
    path = os.path.join(directory, file_stem + "_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)



# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_group(
    ax,
    names: list[str],
    means: list[float],
    errors: list[float],
    colors: list,
    title: str,
    ylabel: str | None,
    hatches: list[str] | None = None,
):
    """Draw a grouped bar chart on *ax* and return the bar objects."""
    x = np.arange(len(names))
    width = 0.6
    bars = ax.bar(
        x,
        means,
        width,
        yerr=errors,
        capsize=4,
        color=colors,
        edgecolor="none",
        hatch=hatches,
        error_kw=dict(zorder=2),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return bars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot LLM vs MuZero comparison")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model_labels", nargs="+", required=True,
                        help="Model labels used when running llm_hanoi_eval.py")
    parser.add_argument("--llm_display_names", nargs="+", default=None,
                        help="Display names for each (model, strategy) pair")
    parser.add_argument("--prompting_strategies", nargs="+",
                        default=["zero_shot", "cot"],
                        help="Prompting strategies to include (must match saved file stems)")
    parser.add_argument("--no_latex", action="store_true",
                        help="Disable LaTeX rendering (use if LaTeX not available)")
    parser.add_argument("--muzero_runs", type=int, default=5,
                        help="Number of MuZero evaluation runs (for SD→SE conversion)")
    args = parser.parse_args()

    set_plot_style()
    if args.no_latex:
        import matplotlib as mpl
        mpl.rcParams["text.usetex"] = False

    root_dir = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp)

    # -----------------------------------------------------------------------
    # Build the condition list for Figure 1 (error)
    # -----------------------------------------------------------------------

    # Assign colours: MuZero conditions use PLOT_COLORS[0..2], LLMs get extras
    extra_colors = ["#8EC8C8", "#D4A5D4", "#F5C281", "#A8D5A2"]
    n_llm = len(args.model_labels) * len(args.prompting_strategies)

    # Build display names: one per (model, strategy) pair
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

    # -----------------------------------------------------------------------
    # Figure 1: MuZero (n_sims=150) vs LLM — mean excess moves
    # Each panel = one difficulty, independent y-axes so bars are readable.
    # Value labels above each bar for clarity.
    # -----------------------------------------------------------------------

    fig_err, axs_err = plt.subplots(
        1, len(DIFFICULTIES),
        figsize=(7.5, 3.5),
        sharey=False,
    )

    # -----------------------------------------------------------------------
    # Figure 2: illegal move rate (LLM only — MuZero doesn't track this)
    # -----------------------------------------------------------------------

    fig_ill, axs_ill = plt.subplots(
        1, len(DIFFICULTIES),
        figsize=(7.5, 3),
        sharey=False,
    )

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)

        err_names, err_means, err_ses, err_colors = [], [], [], []
        ill_names, ill_means, ill_ses, ill_colors = [], [], [], []

        # MuZero conditions at n_sims=150 (last row of accuracy array)
        # Note: column 2 of _actingAccuracy_error.pt is SD (not SE).
        # Convert to SE by dividing by sqrt(muzero_runs).
        muzero_se_factor = 1.0 / np.sqrt(args.muzero_runs)
        for i, (label, name) in enumerate(MUZERO_CONDITIONS):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            mean_e = float(arr[-1, 1])
            sd_e = float(arr[-1, 2]) if arr.shape[1] > 2 else 0.0
            se_e = sd_e * muzero_se_factor
            err_names.append(name)
            err_means.append(mean_e)
            err_ses.append(se_e)
            err_colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

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

                # Error
                err_names.append(disp_name)
                err_means.append(data["mean_error"])
                err_ses.append(data["se_error"])
                err_colors.append(col_c)

                # Illegal move rate
                ill_names.append(disp_name)
                ill_means.append(data["mean_illegal_rate"] * 100)
                ill_ses.append(data["se_illegal_rate"] * 100)
                ill_colors.append(col_c)

        # ---- error bars with value labels ----
        ax_e = axs_err[col]
        bars = _bar_group(
            ax_e,
            err_names,
            err_means,
            err_ses,
            err_colors,
            title=diff_title,
            ylabel="Mean excess moves" if col == 0 else None,
        )
        # Set y-axis with headroom for value labels
        max_val = max(err_means) if err_means else 1.0
        max_bar = max(m + s for m, s in zip(err_means, err_ses)) if err_means else 1.0
        ax_e.set_ylim(bottom=0, top=max_bar * 1.18)
        # Add value labels above error bar tips
        offset = max_val * 0.03
        for bar, val, se in zip(bars, err_means, err_ses):
            ax_e.text(
                bar.get_x() + bar.get_width() / 2,
                val + se + offset,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=6, zorder=5,
            )

        # ---- illegal rate bars with value labels ----
        ax_i = axs_ill[col]
        if ill_means:
            bars_ill = _bar_group(
                ax_i,
                ill_names,
                ill_means,
                ill_ses,
                ill_colors,
                title=diff_title,
                ylabel="Illegal move rate (%)" if col == 0 else None,
            )
            max_ill = max(m + s for m, s in zip(ill_means, ill_ses)) if ill_means else 1.0
            ax_i.set_ylim(bottom=0, top=max_ill * 1.18)
            ill_offset = max(ill_means) * 0.03
            for bar, val, se in zip(bars_ill, ill_means, ill_ses):
                ax_i.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + se + ill_offset,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=6, zorder=5,
                )
        else:
            ax_i.set_visible(False)

    fig_err.tight_layout()
    fig_ill.tight_layout()

    err_path = os.path.join(root_dir, f"LLM_MuZero_ErrorComparison_{args.timestamp}.png")
    ill_path = os.path.join(root_dir, f"LLM_MuZero_IllegalRate_{args.timestamp}.png")
    fig_err.savefig(err_path, dpi=300, bbox_inches="tight")
    fig_ill.savefig(ill_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {err_path}")
    print(f"Saved: {ill_path}")


if __name__ == "__main__":
    main()
