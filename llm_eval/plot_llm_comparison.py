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
    ("Muzero_", "MuZero"),
    ("ResetLatentVal_", "Value ablated\n(PFC lesion)"),
    ("ResetLatentPol_", "Policy ablated\n(Cerebellar)"),
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


def aggregate_from_accuracy(arr: np.ndarray) -> tuple[float, float]:
    """Return (mean_error, se_error) from an accuracy array."""
    mean_e = float(arr[:, 1].mean())
    se_e = float(arr[:, 2].mean()) if arr.shape[1] > 2 else 0.0
    return mean_e, se_e


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
    parser.add_argument("--model_label", required=True,
                        help="Model label used when running llm_hanoi_eval.py")
    parser.add_argument("--llm_display_names", nargs="+",
                        default=["LLM (zero-shot)", "LLM (CoT)"],
                        help="Display names for zero_shot and cot conditions")
    parser.add_argument("--prompting_strategies", nargs="+",
                        default=["zero_shot", "cot"],
                        help="Prompting strategies to include (must match saved file stems)")
    args = parser.parse_args()

    set_plot_style()

    root_dir = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp)

    # -----------------------------------------------------------------------
    # Build the condition list for Figure 1 (error)
    # -----------------------------------------------------------------------

    # Assign colours: MuZero conditions use PLOT_COLORS[0..2], LLMs get extras
    extra_colors = ["#8EC8C8", "#D4A5D4", "#F5C281", "#A8D5A2"]
    llm_colors = extra_colors[: len(args.prompting_strategies)]

    # Determine LLM display names (pad/truncate to match prompting_strategies)
    display_names = list(args.llm_display_names)
    while len(display_names) < len(args.prompting_strategies):
        strat = args.prompting_strategies[len(display_names)]
        display_names.append(f"LLM ({strat})")

    # -----------------------------------------------------------------------
    # Figure 1: mean excess moves per difficulty
    # Figure 2: illegal move rate per difficulty
    # -----------------------------------------------------------------------

    fig_err, axs_err = plt.subplots(
        1, len(DIFFICULTIES),
        figsize=(7.5, 3),
        sharey=True,
    )
    fig_ill, axs_ill = plt.subplots(
        1, len(DIFFICULTIES),
        figsize=(7.5, 3),
        sharey=True,
    )

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)

        # ---- error means & SEs ----
        err_names, err_means, err_ses, err_colors = [], [], [], []
        ill_names, ill_means, ill_ses, ill_colors = [], [], [], []

        # MuZero conditions
        for i, (label, name) in enumerate(MUZERO_CONDITIONS):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            mean_e, se_e = aggregate_from_accuracy(arr)
            err_names.append(name)
            err_means.append(mean_e)
            err_ses.append(se_e)
            err_colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])
            # MuZero doesn't have illegal rate JSON — skip for Figure 2
            # (it tracks error, not illegal rate per episode, in the .pt files)

        # LLM conditions
        for strat, disp_name, col_c in zip(
            args.prompting_strategies, display_names, llm_colors
        ):
            file_stem = f"LLM_{args.model_label}_{strat}"
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
            ill_means.append(data["mean_illegal_rate"] * 100)   # percent
            ill_ses.append(data["se_illegal_rate"] * 100)
            ill_colors.append(col_c)

        # Also load MuZero illegal rate if available (separate json not produced
        # by default — skip gracefully)

        # ---- error bars ----
        ax_e = axs_err[col]
        _bar_group(
            ax_e,
            err_names,
            err_means,
            err_ses,
            err_colors,
            title=diff_title,
            ylabel="Mean excess moves" if col == 0 else None,
        )
        ax_e.set_ylim(bottom=0)

        # ---- illegal rate bars ----
        ax_i = axs_ill[col]
        if ill_means:
            _bar_group(
                ax_i,
                ill_names,
                ill_means,
                ill_ses,
                ill_colors,
                title=diff_title,
                ylabel="Illegal move rate (%)" if col == 0 else None,
            )
            ax_i.set_ylim(bottom=0)
        else:
            ax_i.set_visible(False)

    fig_err.tight_layout()
    fig_ill.tight_layout()

    err_path = os.path.join(root_dir, f"LLM_MuZero_ErrorComparison_{args.timestamp}.png")
    ill_path = os.path.join(root_dir, f"LLM_MuZero_IllegalRate_{args.timestamp}.png")
    fig_err.savefig(err_path, dpi=300)
    fig_ill.savefig(ill_path, dpi=300)
    print(f"Saved: {err_path}")
    print(f"Saved: {ill_path}")

    # -----------------------------------------------------------------------
    # Figure 3: line plot overlay
    # Each panel = one difficulty; MuZero curves + LLM horizontal lines.
    # -----------------------------------------------------------------------

    fig_line, axs_line = plt.subplots(
        1, len(DIFFICULTIES),
        figsize=(7.5, 3),
        sharey=True,
    )

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        ax = axs_line[col]

        # MuZero curves
        for i, (label, name) in enumerate(MUZERO_CONDITIONS):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            errs = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr))
            ax.errorbar(
                arr[:, 0], arr[:, 1], yerr=errs,
                fmt="-o",
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
                linewidth=2,
                markersize=4,
                capsize=3,
                label=name,
            )

        # LLM horizontal lines
        for strat, disp_name, col_c in zip(
            args.prompting_strategies, display_names, llm_colors
        ):
            file_stem = f"LLM_{args.model_label}_{strat}"
            data = load_llm_json(file_dir, file_stem)
            if data is None:
                continue
            mean_e = data["mean_error"]
            se_e = data["se_error"]

            # Find x-range from MuZero data for the span of the hline
            arr_ref = load_accuracy(file_dir, "Muzero_")
            if arr_ref is not None:
                xmin, xmax = arr_ref[:, 0].min(), arr_ref[:, 0].max()
            else:
                xmin, xmax = 0, 150

            ax.hlines(
                mean_e, xmin, xmax,
                colors=col_c,
                linewidths=2,
                linestyles="--",
                label=disp_name,
            )
            ax.fill_between(
                [xmin, xmax],
                mean_e - se_e,
                mean_e + se_e,
                color=col_c,
                alpha=0.15,
            )

        ax.set_title(diff_title)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Excess moves (error)")
        ax.set_xlabel("MCTS simulations per step")
        if col == len(DIFFICULTIES) - 1:
            ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig_line.tight_layout()
    line_path = os.path.join(root_dir, f"LLM_MuZero_LinePlot_{args.timestamp}.png")
    fig_line.savefig(line_path, dpi=300)
    print(f"Saved: {line_path}")


if __name__ == "__main__":
    main()
