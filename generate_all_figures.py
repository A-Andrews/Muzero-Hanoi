#!/usr/bin/env python3
"""Generate all plots and summary tables for the paper.

Single entry point that regenerates every figure and CSV table from the
experimental data.  Run after any new results come in to keep everything
up to date.

Usage:
    python generate_all_figures.py --timestamp 1748875208

    # Without LaTeX (e.g. on a machine without texlive):
    python generate_all_figures.py --timestamp 1748875208 --no_latex
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Project imports ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import PLOT_COLORS, set_plot_style

# ── Constants ─────────────────────────────────────────────────────────────────

MUZERO_CONDITIONS = [
    ("Muzero", "MuZero"),
    ("ResetLatentPol", "Policy Ablated"),
    ("ResetLatentVal", "Value Ablated"),
    ("ResetLatentRwd", "Reward Ablated"),
    ("ResetLatentVal_ResetLatentRwd", "Value + Reward Ablated"),
]

# Subset used for comparison plots (3-condition)
MUZERO_CONDITIONS_SHORT = [
    ("Muzero", "MuZero"),
    ("ResetLatentVal", "Value ablated\n(PFC lesion)"),
    ("ResetLatentPol", "Policy ablated\n(Cerebellar)"),
]

DIFFICULTIES = [
    ("LS", "Close"),
    ("MS", "Moderate"),
    ("ES", "Far"),
]

# For MuZero ablation grid: ES first (top row)
DIFFICULTIES_GRID = [
    ("ES", "Far"),
    ("MS", "Mid"),
    ("LS", "Close"),
]

OPTIMAL_MOVES = {"LS": 1, "MS": 3, "ES": 7}
MAX_STEPS = 200

LLM_MODEL_LABEL = "qwen25_7b"
LLM_PROMPTING_STRATEGIES = ["zero_shot", "cot"]
LLM_DISPLAY_NAMES = ["Qwen-2.5 7B\n(zero-shot)", "Qwen-2.5 7B\n(CoT)"]

LLM_FEEDBACK_STRATEGIES = ["cot", "cot_h5", "cot_h5_illfb"]
LLM_FEEDBACK_DISPLAY = ["CoT", "CoT + horizon", "CoT + horizon\n+ illegal fb"]

LAYER_INDICES = [0, 4, 8, 14, 20, 27]
NOISE_SCALE = 0.5

EXTRA_COLORS = ["#8EC8C8", "#D4A5D4", "#F5C281", "#A8D5A2"]

MUZERO_ILLEGAL_RATES_AGGREGATE = {
    "MuZero": (16.5, 1.5),
    "Policy ablated\n(Cerebellar)": (47.0, 1.5),
    "Value ablated\n(PFC lesion)": (20.5, 1.5),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_accuracy(directory: str, label: str) -> np.ndarray | None:
    """Load *_actingAccuracy(_error).pt → array [n_sims, mean(, std)]."""
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


def muzero_solve_rate(arr: np.ndarray, diff_key: str, muzero_runs: int) -> tuple[float, float]:
    """Estimate solve rate from MuZero error data at n_sims=150."""
    failed_error = MAX_STEPS - OPTIMAL_MOVES[diff_key]
    mean_err = arr[-1, 1]
    sd_err = arr[-1, 2] if arr.shape[1] > 2 else 0.0

    if mean_err + 2 * sd_err < failed_error * 0.8:
        return 1.0, 0.0
    if mean_err - 2 * sd_err > failed_error * 0.8:
        return 0.0, 0.0

    from scipy.stats import norm
    if sd_err > 0:
        solve_prob = norm.cdf(failed_error - 1, loc=mean_err, scale=sd_err)
    else:
        solve_prob = 1.0 if mean_err < failed_error - 1 else 0.0
    se = np.sqrt(solve_prob * (1 - solve_prob) / muzero_runs) if muzero_runs > 1 else 0.0
    return float(solve_prob), float(se)


# ── Bar chart helper ──────────────────────────────────────────────────────────

def _bar_group(ax, names, means, errors, colors, title, ylabel=None,
               value_fmt="{:.1f}", value_fontsize=6, hatches=None):
    x = np.arange(len(names))
    width = 0.6
    bars = ax.bar(
        x, means, width, yerr=errors, capsize=4,
        color=colors, edgecolor="none", hatch=hatches,
        error_kw=dict(zorder=2),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Value labels above bars
    if means:
        max_val = max(means) if means else 1.0
        offset = max_val * 0.03
        for bar, val, se in zip(bars, means, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + se + offset,
                value_fmt.format(val),
                ha="center", va="bottom", fontsize=value_fontsize, zorder=5,
            )
    return bars


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: MuZero ablation grid (error vs simulations)
# ══════════════════════════════════════════════════════════════════════════════

def fig_muzero_ablation_grid(root_dir: str, timestamp: str):
    """3×5 grid: rows=difficulties, cols=ablation conditions."""
    labels = [l for l, _ in MUZERO_CONDITIONS]
    names = [n for _, n in MUZERO_CONDITIONS]

    font_s = 7
    mpl.rc("font", size=font_s)

    fig, axs = plt.subplots(
        nrows=len(DIFFICULTIES_GRID), ncols=len(labels),
        figsize=(7.5, 4),
        gridspec_kw={"wspace": 0.32, "hspace": 0.3},
    )
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.15, top=0.95)

    for e, (d, diff_name) in enumerate(DIFFICULTIES_GRID):
        file_dir = os.path.join(root_dir, d)
        for i, label in enumerate(labels):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            errs = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(arr[:, 1])
            axs[e, i].errorbar(
                arr[:, 0], arr[:, 1], yerr=errs,
                fmt="-o", color=PLOT_COLORS[i % len(PLOT_COLORS)],
                linewidth=2.5, markersize=4, capsize=3,
            )
            axs[e, i].set_ylim([0, 100])
            axs[e, i].spines["right"].set_visible(False)
            axs[e, i].spines["top"].set_visible(False)
            if i != 0:
                axs[e, i].tick_params(axis="y", left=False, labelleft=False)
            if i == 0:
                axs[e, i].set_ylabel(f"{diff_name}\nError", fontsize=font_s)
            if e == 0:
                axs[e, i].set_title(names[i], fontsize=font_s)
            if e == len(DIFFICULTIES_GRID) - 1:
                axs[e, i].set_xlabel("N. simulations per step\n(planning time)", fontsize=font_s)

    out = os.path.join(root_dir, f"MuZero_Ablation_Comparison_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: MuZero bar charts (simulations to baseline)
# ══════════════════════════════════════════════════════════════════════════════

def fig_muzero_bar_charts(root_dir: str, timestamp: str):
    labels = [l for l, _ in MUZERO_CONDITIONS_SHORT]
    names = [n.replace("\n", " ") for _, n in MUZERO_CONDITIONS_SHORT]
    state_titles = ["Close to goal", "Mid distance", "Far from goal"]
    directories_bar = ["LS", "MS", "ES"]

    font_s = 7
    mpl.rc("font", size=font_s)

    fig_bar, axs_bar = plt.subplots(1, len(directories_bar), figsize=(7.5, 3), sharey=True)

    for e, d in enumerate(directories_bar):
        file_dir = os.path.join(root_dir, d)
        results = [load_accuracy(file_dir, l) for l in labels]
        if results[0] is None:
            continue
        mu_zero_avg = results[0][:, 1].mean()

        times_to_reach, time_errs, never_reached_mask = [], [], []
        for r in results:
            if r is None:
                times_to_reach.append(0)
                time_errs.append(0)
                never_reached_mask.append(True)
                continue
            indices = np.where(r[:, 1] <= mu_zero_avg)[0]
            if len(indices) > 0:
                times_to_reach.append(r[indices[0], 0])
                time_errs.append(r[indices[0], 2] if r.shape[1] > 2 else 0)
                never_reached_mask.append(False)
            else:
                times_to_reach.append(r[-1, 0])
                time_errs.append(r[-1, 2] if r.shape[1] > 2 else 0)
                never_reached_mask.append(True)

        bars = axs_bar[e].bar(
            names, times_to_reach, yerr=time_errs, capsize=5,
            color=[PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(names))],
            edgecolor="none",
        )
        max_height = max(times_to_reach) if times_to_reach else 1
        axs_bar[e].set_ylim(0, max_height * 1.25)
        axs_bar[e].set_title(state_titles[e], pad=10)
        if e == 0:
            axs_bar[e].set_ylabel("Simulations to\nbase rate")
        axs_bar[e].spines["right"].set_visible(False)
        axs_bar[e].spines["top"].set_visible(False)
        if e == len(directories_bar) - 1:
            axs_bar[e].legend(bars, names, fontsize=font_s, bbox_to_anchor=(1.05, 1), loc="upper right")
        axs_bar[e].get_xaxis().set_visible(False)

        for idx, (bar, never) in enumerate(zip(bars, never_reached_mask)):
            if never:
                bar.set_hatch("//")
                axs_bar[e].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_height * 0.05,
                    "*", ha="center", va="bottom", color="red", fontweight="bold",
                )

    fig_bar.tight_layout()
    out = os.path.join(root_dir, f"MuZero_Ablation_BarCharts_{timestamp}.png")
    fig_bar.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MuZero average performance
# ══════════════════════════════════════════════════════════════════════════════

def fig_muzero_average_performance(root_dir: str, timestamp: str):
    labels = [l for l, _ in MUZERO_CONDITIONS_SHORT]
    names = [n.replace("\n", " ") for _, n in MUZERO_CONDITIONS_SHORT]
    state_titles = ["Close to goal", "Mid distance", "Far from goal"]
    directories_bar = ["LS", "MS", "ES"]

    font_s = 7
    mpl.rc("font", size=font_s)

    fig_avg, axs_avg = plt.subplots(1, len(directories_bar), figsize=(7.5, 3), sharey=True)

    for e, d in enumerate(directories_bar):
        file_dir = os.path.join(root_dir, d)
        means, errs = [], []
        for l in labels:
            arr = load_accuracy(file_dir, l)
            if arr is None:
                means.append(0)
                errs.append(0)
                continue
            means.append(arr[:, 1].mean())
            errs.append(arr[:, 2].mean() if arr.shape[1] > 2 else 0)

        bars = axs_avg[e].bar(
            names, means, yerr=errs, capsize=5,
            color=[PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(names))],
            edgecolor="none",
        )
        axs_avg[e].get_xaxis().set_visible(False)
        axs_avg[e].set_title(state_titles[e])
        if e == 0:
            axs_avg[e].set_ylabel("Mean Error")
        axs_avg[e].spines["right"].set_visible(False)
        axs_avg[e].spines["top"].set_visible(False)
        if e == len(directories_bar) - 1:
            axs_avg[e].legend(bars, names, fontsize=font_s, bbox_to_anchor=(1.05, 1), loc="upper right")

    fig_avg.tight_layout()
    out = os.path.join(root_dir, f"MuZero_Ablation_AveragePerformance_{timestamp}.png")
    fig_avg.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_avg)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: LLM vs MuZero — error comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_llm_muzero_error(root_dir: str, timestamp: str, muzero_runs: int):
    """Bar chart: MuZero (n_sims=150) vs LLM prompting strategies — mean error."""
    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=False)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, means, ses, colors = [], [], [], []

        # MuZero conditions
        for i, (label, name) in enumerate(MUZERO_CONDITIONS_SHORT):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            mean_e = float(arr[-1, 1])
            sd_e = float(arr[-1, 2]) if arr.shape[1] > 2 else 0.0
            names.append(name)
            means.append(mean_e)
            ses.append(sd_e * se_factor)
            colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

        # LLM conditions
        for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
            file_stem = f"LLM_{LLM_MODEL_LABEL}_{strat}"
            data = load_llm_json(file_dir, file_stem)
            if data is None:
                continue
            names.append(LLM_DISPLAY_NAMES[j])
            means.append(data["mean_error"])
            ses.append(data["se_error"])
            colors.append(EXTRA_COLORS[j % len(EXTRA_COLORS)])

        ax = axs[col]
        _bar_group(ax, names, means, ses, colors, diff_title,
                   ylabel="Mean excess moves" if col == 0 else None)
        if means:
            max_bar = max(m + s for m, s in zip(means, ses))
            ax.set_ylim(bottom=0, top=max_bar * 1.18)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_MuZero_ErrorComparison_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Solve rate comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_solve_rate(root_dir: str, timestamp: str, muzero_runs: int):
    font_s = 7
    mpl.rc("font", size=font_s)

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=True)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        for i, (label, name) in enumerate(MUZERO_CONDITIONS_SHORT):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            sr, se = muzero_solve_rate(arr, diff_dir, muzero_runs)
            names.append(name)
            rates.append(sr * 100)
            ses.append(se * 100)
            colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

        for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
            file_stem = f"LLM_{LLM_MODEL_LABEL}_{strat}"
            data = load_llm_json(file_dir, file_stem)
            if data is None:
                continue
            names.append(LLM_DISPLAY_NAMES[j])
            rates.append(data["solve_rate"] * 100)
            ses.append(data.get("se_solve_rate", 0.0) * 100)
            colors.append(EXTRA_COLORS[j % len(EXTRA_COLORS)])

        ax = axs[col]
        _bar_group(ax, names, rates, ses, colors, diff_title,
                   ylabel="Solve rate (%)" if col == 0 else None,
                   value_fmt="{:.0f}%")
        ax.set_ylim(0, 115)

    fig.tight_layout()
    out = os.path.join(root_dir, f"SolveRate_Comparison_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Illegal move rate comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_illegal_rate(root_dir: str, timestamp: str):
    font_s = 7
    mpl.rc("font", size=font_s)

    muzero_conds = [
        ("MuZero", PLOT_COLORS[0]),
        ("Value ablated\n(PFC lesion)", PLOT_COLORS[2]),
        ("Policy ablated\n(Cerebellar)", PLOT_COLORS[1]),
    ]

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=True)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        for cond_name, cond_color in muzero_conds:
            rate, se = MUZERO_ILLEGAL_RATES_AGGREGATE[cond_name]
            names.append(cond_name)
            rates.append(rate)
            ses.append(se)
            colors.append(cond_color)

        for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
            file_stem = f"LLM_{LLM_MODEL_LABEL}_{strat}"
            data = load_llm_json(file_dir, file_stem)
            if data is None:
                continue
            names.append(LLM_DISPLAY_NAMES[j])
            rates.append(data["mean_illegal_rate"] * 100)
            ses.append(data["se_illegal_rate"] * 100)
            colors.append(EXTRA_COLORS[j % len(EXTRA_COLORS)])

        ax = axs[col]
        _bar_group(ax, names, rates, ses, colors, diff_title,
                   ylabel="Illegal move rate (%)" if col == 0 else None,
                   value_fmt="{:.1f}%")
        if rates:
            max_bar = max(r + s for r, s in zip(rates, ses))
            ax.set_ylim(0, max_bar * 1.2)

    fig.text(0.5, -0.02, "(MuZero: aggregate over random starts, n\\_sims=25)",
             ha="center", fontsize=6, style="italic")
    fig.tight_layout()
    out = os.path.join(root_dir, f"IllegalRate_Comparison_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: LLM feedback sweep (CoT → +horizon → +illegal feedback)
# ══════════════════════════════════════════════════════════════════════════════

def fig_feedback_sweep(root_dir: str, timestamp: str):
    """Bar chart showing how error drops as more feedback is provided."""
    font_s = 7
    mpl.rc("font", size=font_s)
    fb_colors = ["#8EC8C8", "#D4A5D4", "#F5C281"]

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(7.5, 3.5), sharey=False)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, means, ses, colors = [], [], [], []

        for j, strat in enumerate(LLM_FEEDBACK_STRATEGIES):
            file_stem = f"LLM_{LLM_MODEL_LABEL}_{strat}"
            data = load_llm_json(file_dir, file_stem)
            if data is None:
                continue
            names.append(LLM_FEEDBACK_DISPLAY[j])
            means.append(data["mean_error"])
            ses.append(data["se_error"])
            colors.append(fb_colors[j % len(fb_colors)])

        ax = axs[col]
        _bar_group(ax, names, means, ses, colors, diff_title,
                   ylabel="Mean excess moves" if col == 0 else None)
        if means:
            max_bar = max(m + s for m, s in zip(means, ses))
            ax.set_ylim(bottom=0, top=max_bar * 1.25)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_FeedbackSweep_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: LLM layer ablation (error vs layer index) — all difficulties
# ══════════════════════════════════════════════════════════════════════════════

def fig_layer_ablation(root_dir: str, timestamp: str):
    """One panel per difficulty showing ablation + noise curves."""
    font_s = 8
    mpl.rc("font", size=font_s)

    color_ablation = PLOT_COLORS[2]
    color_noise = PLOT_COLORS[1]
    color_baseline = PLOT_COLORS[0]

    diff_name_map = {"ES": "Far (7 moves)", "MS": "Moderate (3 moves)", "LS": "Close (1 move)"}

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)

    for col, (diff_dir, _) in enumerate(DIFFICULTIES):
        diff_root = os.path.join(root_dir, diff_dir)
        ax = axs[col]

        # Baseline
        baseline_stem = f"LLM_{LLM_MODEL_LABEL}_cot"
        baseline_data = load_llm_json(diff_root, baseline_stem)

        # Ablation sweep
        abl_means, abl_ses, abl_layers = [], [], []
        for layer in LAYER_INDICES:
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_ablateL{layer}")
            if res is not None:
                abl_means.append(res["mean_error"])
                abl_ses.append(res["se_error"])
                abl_layers.append(layer)

        # Noise sweep
        noi_means, noi_ses, noi_layers = [], [], []
        for layer in LAYER_INDICES:
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_noiseS{NOISE_SCALE}_L{layer}")
            if res is not None:
                noi_means.append(res["mean_error"])
                noi_ses.append(res["se_error"])
                noi_layers.append(layer)

        # Baseline line
        if baseline_data is not None:
            ax.axhline(baseline_data["mean_error"], color=color_baseline, linestyle="--",
                       linewidth=1.5, label=f"No intervention ({baseline_data['mean_error']:.1f})", zorder=1)
            ax.fill_between(
                [min(LAYER_INDICES) - 1, max(LAYER_INDICES) + 1],
                baseline_data["mean_error"] - baseline_data["se_error"],
                baseline_data["mean_error"] + baseline_data["se_error"],
                color=color_baseline, alpha=0.1, zorder=0,
            )

        if abl_layers:
            ax.errorbar(abl_layers, abl_means, yerr=abl_ses, fmt="-o",
                        color=color_ablation, linewidth=2.0, markersize=5, capsize=3,
                        label="Layer ablation (skip)", zorder=3)

        if noi_layers:
            use_latex = mpl.rcParams.get("text.usetex", False)
            sigma_label = f"Noise injection ($\\sigma={NOISE_SCALE}$)" if use_latex else f"Noise injection (σ={NOISE_SCALE})"
            ax.errorbar(noi_layers, noi_means, yerr=noi_ses, fmt="-s",
                        color=color_noise, linewidth=2.0, markersize=5, capsize=3,
                        label=sigma_label, zorder=2)

        ax.set_xlabel("Transformer layer index", fontsize=font_s + 1)
        if col == 0:
            ax.set_ylabel("Mean error (steps above optimal)", fontsize=font_s + 1)
        ax.set_title(diff_name_map[diff_dir], fontsize=font_s + 2)
        ax.set_xticks(LAYER_INDICES)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.legend(fontsize=font_s - 1, frameon=False)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_LayerSweep_AllDifficulties_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # Also save per-difficulty versions
    for diff_dir, _ in DIFFICULTIES:
        diff_root = os.path.join(root_dir, diff_dir)
        baseline_data = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot")

        abl_means, abl_ses, abl_layers = [], [], []
        noi_means, noi_ses, noi_layers = [], [], []
        for layer in LAYER_INDICES:
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_ablateL{layer}")
            if res is not None:
                abl_means.append(res["mean_error"])
                abl_ses.append(res["se_error"])
                abl_layers.append(layer)
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_noiseS{NOISE_SCALE}_L{layer}")
            if res is not None:
                noi_means.append(res["mean_error"])
                noi_ses.append(res["se_error"])
                noi_layers.append(layer)

        if not abl_layers and not noi_layers:
            continue

        fig_s, ax_s = plt.subplots(figsize=(5.5, 3.5))
        if baseline_data is not None:
            ax_s.axhline(baseline_data["mean_error"], color=color_baseline, linestyle="--",
                         linewidth=1.5, label=f"No intervention ({baseline_data['mean_error']:.1f})", zorder=1)
            ax_s.fill_between(
                [min(LAYER_INDICES) - 1, max(LAYER_INDICES) + 1],
                baseline_data["mean_error"] - baseline_data["se_error"],
                baseline_data["mean_error"] + baseline_data["se_error"],
                color=color_baseline, alpha=0.1, zorder=0,
            )
        if abl_layers:
            ax_s.errorbar(abl_layers, abl_means, yerr=abl_ses, fmt="-o",
                          color=color_ablation, linewidth=2.0, markersize=5, capsize=3,
                          label="Layer ablation (skip)", zorder=3)
        if noi_layers:
            use_latex = mpl.rcParams.get("text.usetex", False)
            sigma_label = f"Noise injection ($\\sigma={NOISE_SCALE}$)" if use_latex else f"Noise injection (σ={NOISE_SCALE})"
            ax_s.errorbar(noi_layers, noi_means, yerr=noi_ses, fmt="-s",
                          color=color_noise, linewidth=2.0, markersize=5, capsize=3,
                          label=sigma_label, zorder=2)
        ax_s.set_xlabel("Transformer layer index")
        ax_s.set_ylabel("Mean error (steps above optimal)")
        ax_s.set_title(f"LLM layerwise intervention — {diff_name_map[diff_dir]} (CoT)")
        ax_s.set_xticks(LAYER_INDICES)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)
        ax_s.legend(fontsize=font_s, frameon=False)
        fig_s.tight_layout()
        out_s = os.path.join(diff_root, f"LLM_LayerSweep_{timestamp}.png")
        fig_s.savefig(out_s, dpi=300, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Saved: {out_s}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: LLM ablation matrix (mirrors MuZero ablation grid)
# ══════════════════════════════════════════════════════════════════════════════

def fig_llm_ablation_matrix(root_dir: str, timestamp: str):
    """3×2 grid: rows = difficulties (Far/Mid/Close), cols = (layer ablation, noise).

    Baseline CoT shown as dashed horizontal line in each cell.
    Directly parallels the MuZero ablation grid (Fig 1).
    """
    font_s = 7
    mpl.rc("font", size=font_s)

    color_ablation = PLOT_COLORS[2]   # red-ish
    color_noise = PLOT_COLORS[1]      # blue-ish
    color_baseline = PLOT_COLORS[0]   # black

    col_titles = ["Layer Ablation (skip)", "Noise Injection"]
    diff_name_map = {"ES": "Far", "MS": "Mid", "LS": "Close"}

    fig, axs = plt.subplots(
        nrows=3, ncols=2, figsize=(7.5, 5.5),
        gridspec_kw={"wspace": 0.25, "hspace": 0.35},
    )

    for row, (diff_dir, _) in enumerate(DIFFICULTIES_GRID):
        diff_root = os.path.join(root_dir, diff_dir)

        # Baseline
        baseline_data = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot")

        # Ablation sweep
        abl_means, abl_ses, abl_layers = [], [], []
        for layer in LAYER_INDICES:
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_ablateL{layer}")
            if res is not None:
                abl_means.append(res["mean_error"])
                abl_ses.append(res["se_error"])
                abl_layers.append(layer)

        # Noise sweep
        noi_means, noi_ses, noi_layers = [], [], []
        for layer in LAYER_INDICES:
            res = load_llm_json(diff_root, f"LLM_{LLM_MODEL_LABEL}_cot_noiseS{NOISE_SCALE}_L{layer}")
            if res is not None:
                noi_means.append(res["mean_error"])
                noi_ses.append(res["se_error"])
                noi_layers.append(layer)

        sweeps = [
            (abl_layers, abl_means, abl_ses, color_ablation, "-o"),
            (noi_layers, noi_means, noi_ses, color_noise, "-s"),
        ]

        for col_idx, (layers, means, ses, color, fmt) in enumerate(sweeps):
            ax = axs[row, col_idx]

            # Baseline dashed line + shaded SE band
            if baseline_data is not None:
                ax.axhline(baseline_data["mean_error"], color=color_baseline,
                           linestyle="--", linewidth=1.5, zorder=1)
                ax.fill_between(
                    [min(LAYER_INDICES) - 1, max(LAYER_INDICES) + 1],
                    baseline_data["mean_error"] - baseline_data["se_error"],
                    baseline_data["mean_error"] + baseline_data["se_error"],
                    color=color_baseline, alpha=0.1, zorder=0,
                )

            # Sweep curve
            if layers:
                ax.errorbar(layers, means, yerr=ses, fmt=fmt,
                            color=color, linewidth=2.0, markersize=4, capsize=3, zorder=3)

            ax.set_xticks(LAYER_INDICES)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(f"{diff_name_map[diff_dir]}\nError", fontsize=font_s)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)

            # Column titles (top row only)
            if row == 0:
                ax.set_title(col_titles[col_idx], fontsize=font_s)

            # X-axis label (bottom row only)
            if row == 2:
                ax.set_xlabel("Transformer layer index", fontsize=font_s)

    out = os.path.join(root_dir, f"LLM_AblationMatrix_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: MuZero heatmap (conditions × difficulties)
# ══════════════════════════════════════════════════════════════════════════════

MUZERO_HEATMAP_CONDITIONS = [
    ("Muzero", "None (baseline)"),
    ("ResetLatentPol", "Policy"),
    ("ResetLatentVal", "Value"),
    ("ResetLatentRwd", "Reward"),
    ("ResetLatentPol_ResetLatentVal", "Policy + Value"),
    ("ResetLatentPol_ResetLatentRwd", "Policy + Reward"),
    ("ResetLatentVal_ResetLatentRwd", "Value + Reward"),
    ("ResetLatentPol_ResetLatentVal_ResetLatentRwd", "Policy + Value + Reward"),
]

HEATMAP_DIFFICULTIES = [
    ("ES", "Far (7 moves)"),
    ("MS", "Mid (3 moves)"),
    ("LS", "Close (1 move)"),
]


def _draw_heatmap(ax, data, se_data, row_labels, col_labels, title, cbar_label,
                  vmin=0, vmax=None):
    """Draw a heatmap with annotated values on *ax*."""
    from matplotlib.colors import TwoSlopeNorm

    if vmax is None:
        vmax = np.nanmax(data)

    # Diverging colormap centred at a low value to highlight bad performance
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmax * 0.3, vmax=vmax)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Starting distance from goal", fontsize=8)
    ax.set_title(title, fontsize=9, pad=10)

    # Annotate each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            se = se_data[i, j] if se_data is not None else None
            if np.isnan(val):
                continue
            # Choose text colour for contrast
            text_color = "white" if val > vmax * 0.55 else "black"
            if se is not None and not np.isnan(se) and se > 0:
                txt = f"{val:.1f}\n$\\pm${se:.1f}" if mpl.rcParams.get("text.usetex", False) else f"{val:.1f}\n±{se:.1f}"
            else:
                txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color=text_color, fontweight="bold")

    return im


def fig_muzero_heatmap(root_dir: str, timestamp: str, muzero_runs: int):
    """Heatmap: MuZero ablation conditions (rows) × difficulties (cols)."""
    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

    row_labels = [name for _, name in MUZERO_HEATMAP_CONDITIONS]
    col_labels = [name for _, name in HEATMAP_DIFFICULTIES]

    n_rows = len(MUZERO_HEATMAP_CONDITIONS)
    n_cols = len(HEATMAP_DIFFICULTIES)
    data = np.full((n_rows, n_cols), np.nan)
    se_data = np.full((n_rows, n_cols), np.nan)

    for j, (diff_dir, _) in enumerate(HEATMAP_DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        for i, (label, _) in enumerate(MUZERO_HEATMAP_CONDITIONS):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            data[i, j] = arr[-1, 1]  # n_sims=150
            if arr.shape[1] > 2:
                se_data[i, j] = arr[-1, 2] * se_factor

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = _draw_heatmap(ax, data, se_data, row_labels, col_labels,
                       title="Mean error at n\\_sims=150" if mpl.rcParams.get("text.usetex", False) else "Mean error at n_sims=150",
                       cbar_label="Mean error (steps above optimal)",
                       vmin=0, vmax=200)
    ax.set_ylabel("Ablated component(s)", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean error (steps above optimal)", fontsize=7)

    fig.tight_layout()
    out = os.path.join(root_dir, f"MuZero_AblationMatrix_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: LLM heatmap (conditions × difficulties)
# ══════════════════════════════════════════════════════════════════════════════

LLM_HEATMAP_CONDITIONS = [
    (f"LLM_{LLM_MODEL_LABEL}_zero_shot", "Zero-shot"),
    (f"LLM_{LLM_MODEL_LABEL}_cot", "CoT"),
    (f"LLM_{LLM_MODEL_LABEL}_cot_h5", "CoT + horizon"),
    (f"LLM_{LLM_MODEL_LABEL}_cot_h5_illfb", "CoT + horizon + illegal fb"),
] + [
    (f"LLM_{LLM_MODEL_LABEL}_cot_ablateL{l}", f"CoT + ablate layer {l}")
    for l in LAYER_INDICES
] + [
    (f"LLM_{LLM_MODEL_LABEL}_cot_noiseS{NOISE_SCALE}_L{l}", f"CoT + noise layer {l}")
    for l in LAYER_INDICES
]


def fig_llm_heatmap(root_dir: str, timestamp: str):
    """Heatmap: LLM conditions (rows) × difficulties (cols)."""
    font_s = 7
    mpl.rc("font", size=font_s)

    col_labels = [name for _, name in HEATMAP_DIFFICULTIES]

    # Filter to conditions that actually have data
    all_stems = []
    all_names = []
    for stem, name in LLM_HEATMAP_CONDITIONS:
        # Check if at least one difficulty has data
        for diff_dir, _ in HEATMAP_DIFFICULTIES:
            path = os.path.join(root_dir, diff_dir, stem + "_results.json")
            if os.path.exists(path):
                all_stems.append(stem)
                all_names.append(name)
                break

    n_rows = len(all_stems)
    n_cols = len(HEATMAP_DIFFICULTIES)
    data = np.full((n_rows, n_cols), np.nan)
    se_data = np.full((n_rows, n_cols), np.nan)

    for j, (diff_dir, _) in enumerate(HEATMAP_DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        for i, stem in enumerate(all_stems):
            res = load_llm_json(file_dir, stem)
            if res is not None:
                data[i, j] = res["mean_error"]
                se_data[i, j] = res["se_error"]

    fig, ax = plt.subplots(figsize=(5.5, 0.4 * n_rows + 1.8))
    im = _draw_heatmap(ax, data, se_data, all_names, col_labels,
                       title="LLM mean error by condition",
                       cbar_label="Mean error (steps above optimal)",
                       vmin=0, vmax=200)
    ax.set_ylabel("LLM condition", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean error (steps above optimal)", fontsize=7)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_ConditionMatrix_{timestamp}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLES: CSV summaries
# ══════════════════════════════════════════════════════════════════════════════

def generate_tables(root_dir: str, muzero_runs: int):
    se_factor = 1.0 / math.sqrt(muzero_runs)
    table_dir = os.path.join(root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # ── muzero_summary.csv ──
    rows = []
    for diff_dir, diff_name in DIFFICULTIES:
        file_dir = os.path.join(root_dir, diff_dir)
        for label, display_name in MUZERO_CONDITIONS:
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            has_std = arr.shape[1] > 2
            for row in arr:
                n_sims = int(row[0])
                mean_error = row[1]
                std_error = row[2] if has_std else float("nan")
                se_error = std_error * se_factor if has_std else float("nan")
                rows.append({
                    "condition": label, "display_name": display_name,
                    "difficulty": diff_name, "difficulty_code": diff_dir,
                    "n_sims": n_sims,
                    "mean_error": f"{mean_error:.4f}",
                    "std_error": f"{std_error:.4f}" if has_std else "",
                    "se_error": f"{se_error:.4f}" if has_std else "",
                })

    fields = ["condition", "display_name", "difficulty", "difficulty_code",
              "n_sims", "mean_error", "std_error", "se_error"]
    p = os.path.join(table_dir, "muzero_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {p} ({len(rows)} rows)")

    # ── llm_summary.csv ──
    rows = []
    for diff_dir, diff_name in DIFFICULTIES:
        file_dir = os.path.join(root_dir, diff_dir)
        for json_path in sorted(glob.glob(os.path.join(file_dir, "LLM_*_results.json"))):
            fname = os.path.basename(json_path)
            condition = fname.replace("LLM_", "").replace("_results.json", "")
            with open(json_path) as f:
                data = json.load(f)
            rows.append({
                "condition": condition, "difficulty": diff_name,
                "difficulty_code": diff_dir,
                "mean_error": f"{data['mean_error']:.4f}",
                "se_error": f"{data['se_error']:.4f}",
                "solve_rate": f"{data['solve_rate']:.4f}",
                "se_solve_rate": f"{data.get('se_solve_rate', 0):.4f}",
                "illegal_rate": f"{data['mean_illegal_rate']:.4f}",
                "se_illegal_rate": f"{data['se_illegal_rate']:.4f}",
                "n_episodes": data.get("n_episodes", ""),
            })

    fields = ["condition", "difficulty", "difficulty_code", "mean_error",
              "se_error", "solve_rate", "se_solve_rate", "illegal_rate",
              "se_illegal_rate", "n_episodes"]
    p = os.path.join(table_dir, "llm_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {p} ({len(rows)} rows)")

    # ── combined_summary.csv ──
    rows = []
    for diff_dir, diff_name in DIFFICULTIES:
        file_dir = os.path.join(root_dir, diff_dir)

        for label, display_name in MUZERO_CONDITIONS:
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            has_std = arr.shape[1] > 2
            last = arr[-1]
            rows.append({
                "agent_type": "MuZero", "condition": display_name,
                "difficulty": diff_name, "difficulty_code": diff_dir,
                "mean_error": f"{last[1]:.4f}",
                "se_error": f"{last[2] * se_factor:.4f}" if has_std else "",
                "solve_rate": "", "se_solve_rate": "",
                "illegal_rate": "", "se_illegal_rate": "",
            })

        for json_path in sorted(glob.glob(os.path.join(file_dir, "LLM_*_results.json"))):
            fname = os.path.basename(json_path)
            condition = fname.replace("LLM_", "").replace("_results.json", "")
            with open(json_path) as f:
                data = json.load(f)
            rows.append({
                "agent_type": "LLM", "condition": condition,
                "difficulty": diff_name, "difficulty_code": diff_dir,
                "mean_error": f"{data['mean_error']:.4f}",
                "se_error": f"{data['se_error']:.4f}",
                "solve_rate": f"{data['solve_rate']:.4f}",
                "se_solve_rate": f"{data.get('se_solve_rate', 0):.4f}",
                "illegal_rate": f"{data['mean_illegal_rate']:.4f}",
                "se_illegal_rate": f"{data['se_illegal_rate']:.4f}",
            })

    fields = ["agent_type", "condition", "difficulty", "difficulty_code",
              "mean_error", "se_error", "solve_rate", "se_solve_rate",
              "illegal_rate", "se_illegal_rate"]
    p = os.path.join(table_dir, "combined_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {p} ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and summary tables"
    )
    parser.add_argument("--timestamp", required=True,
                        help="Experiment timestamp directory")
    parser.add_argument("--no_latex", action="store_true",
                        help="Disable LaTeX rendering")
    parser.add_argument("--muzero_runs", type=int, default=5,
                        help="Number of MuZero runs for SD→SE conversion")
    args = parser.parse_args()

    root_dir = os.path.join("stats", "Hanoi", args.timestamp)
    if not os.path.isdir(root_dir):
        print(f"Error: directory not found: {root_dir}")
        return

    set_plot_style()
    if args.no_latex:
        mpl.rcParams["text.usetex"] = False

    print(f"Generating all figures for timestamp {args.timestamp}")
    print(f"Output directory: {root_dir}\n")

    print("[1/11] MuZero ablation grid")
    fig_muzero_ablation_grid(root_dir, args.timestamp)

    print("[2/11] MuZero bar charts (simulations to baseline)")
    fig_muzero_bar_charts(root_dir, args.timestamp)

    print("[3/11] MuZero average performance")
    fig_muzero_average_performance(root_dir, args.timestamp)

    print("[4/11] LLM vs MuZero — error comparison")
    fig_llm_muzero_error(root_dir, args.timestamp, args.muzero_runs)

    print("[5/11] Solve rate comparison")
    fig_solve_rate(root_dir, args.timestamp, args.muzero_runs)

    print("[6/11] Illegal move rate comparison")
    fig_illegal_rate(root_dir, args.timestamp)

    print("[7/11] LLM feedback sweep")
    fig_feedback_sweep(root_dir, args.timestamp)

    print("[8/11] LLM layer ablation")
    fig_layer_ablation(root_dir, args.timestamp)

    print("[9/11] LLM ablation matrix")
    fig_llm_ablation_matrix(root_dir, args.timestamp)

    print("[10/11] MuZero heatmap")
    fig_muzero_heatmap(root_dir, args.timestamp, args.muzero_runs)

    print("[11/11] LLM heatmap")
    fig_llm_heatmap(root_dir, args.timestamp)

    print("\n[Tables] Generating CSV summaries")
    generate_tables(root_dir, args.muzero_runs)

    print("\nDone! All figures and tables regenerated.")


if __name__ == "__main__":
    main()
