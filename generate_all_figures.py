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

# Both LLM models for cross-model comparison figures
LLM_MODELS = [
    ("qwen25_7b", "Qwen-2.5 7B"),
    ("llama3_8b", "Llama-3.1 8B"),
]
LLM_MODEL_COLORS = {
    "qwen25_7b": ["#8EC8C8", "#D4A5D4"],   # zero-shot, cot
    "llama3_8b": ["#F5C281", "#A8D5A2"],    # zero-shot, cot
}

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
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
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

def fig_muzero_ablation_grid(root_dir: str, timestamp: str, muzero_runs: int = 5):
    """3×5 grid: rows=difficulties, cols=ablation conditions."""
    labels = [l for l, _ in MUZERO_CONDITIONS]
    names = [n for _, n in MUZERO_CONDITIONS]

    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

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
            errs = arr[:, 2] * se_factor if arr.shape[1] > 2 else np.zeros_like(arr[:, 1])
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

    out = os.path.join(root_dir, f"MuZero_Ablation_Comparison_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: MuZero bar charts (simulations to baseline)
# ══════════════════════════════════════════════════════════════════════════════

def fig_muzero_bar_charts(root_dir: str, timestamp: str, muzero_runs: int = 5):
    labels = [l for l, _ in MUZERO_CONDITIONS_SHORT]
    names = [n.replace("\n", " ") for _, n in MUZERO_CONDITIONS_SHORT]
    state_titles = ["Close to goal", "Mid distance", "Far from goal"]
    directories_bar = ["LS", "MS", "ES"]

    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

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
                time_errs.append(r[indices[0], 2] * se_factor if r.shape[1] > 2 else 0)
                never_reached_mask.append(False)
            else:
                times_to_reach.append(r[-1, 0])
                time_errs.append(r[-1, 2] * se_factor if r.shape[1] > 2 else 0)
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
    out = os.path.join(root_dir, f"MuZero_Ablation_BarCharts_{timestamp}.pdf")
    fig_bar.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MuZero average performance
# ══════════════════════════════════════════════════════════════════════════════

def fig_muzero_average_performance(root_dir: str, timestamp: str, muzero_runs: int = 5):
    labels = [l for l, _ in MUZERO_CONDITIONS_SHORT]
    names = [n.replace("\n", " ") for _, n in MUZERO_CONDITIONS_SHORT]
    state_titles = ["Close to goal", "Mid distance", "Far from goal"]
    directories_bar = ["LS", "MS", "ES"]

    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

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
            errs.append(float(arr[-1, 2] * se_factor) if arr.shape[1] > 2 else 0.0)

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
    out = os.path.join(root_dir, f"MuZero_Ablation_AveragePerformance_{timestamp}.pdf")
    fig_avg.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_avg)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: LLM vs MuZero — error comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_llm_muzero_error(root_dir: str, timestamp: str, muzero_runs: int):
    """Bar chart: MuZero (n_sims=150) vs LLM prompting strategies — mean error.

    Includes both Qwen and Llama (if data exists).
    """
    font_s = 7
    mpl.rc("font", size=font_s)
    se_factor = 1.0 / np.sqrt(muzero_runs)

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(12, 3.5), sharey=False)

    # Short display names for MuZero conditions
    muzero_short_labels = {
        "MuZero": "MuZero",
        "Value ablated\n(PFC lesion)": "Value abl.\n(PFC)",
        "Policy ablated\n(Cerebellar)": "Policy abl.\n(Cereb.)",
    }
    # Short model names for LLM bars
    llm_short_names = {
        "qwen25_7b": "Qwen",
        "llama3_8b": "Llama",
    }

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
            names.append(muzero_short_labels.get(name, name))
            means.append(mean_e)
            ses.append(sd_e * se_factor)
            colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

        # LLM conditions — both models
        for model_label, model_display in LLM_MODELS:
            model_colors = LLM_MODEL_COLORS[model_label]
            short_name = llm_short_names.get(model_label, model_display)
            for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
                file_stem = f"LLM_{model_label}_{strat}"
                data = load_llm_json(file_dir, file_stem)
                if data is None:
                    continue
                strat_display = "0-shot" if strat == "zero_shot" else strat
                names.append(f"{short_name}\n({strat_display})")
                means.append(data["mean_error"])
                ses.append(data["se_error"])
                colors.append(model_colors[j % len(model_colors)])

        ax = axs[col]
        _bar_group(ax, names, means, ses, colors, diff_title,
                   ylabel="Mean excess moves" if col == 0 else None)
        if means:
            max_bar = max(m + s for m, s in zip(means, ses))
            ax.set_ylim(bottom=0, top=max_bar * 1.18)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_MuZero_ErrorComparison_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Solve rate comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_solve_rate(root_dir: str, timestamp: str, muzero_runs: int):
    font_s = 7
    mpl.rc("font", size=font_s)

    # Short labels (same as fig_llm_muzero_error)
    muzero_short_labels = {
        "MuZero": "MuZero",
        "Value ablated\n(PFC lesion)": "Value abl.\n(PFC)",
        "Policy ablated\n(Cerebellar)": "Policy abl.\n(Cereb.)",
    }
    llm_short_names = {"qwen25_7b": "Qwen", "llama3_8b": "Llama"}

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(12, 3.5), sharey=True)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        for i, (label, name) in enumerate(MUZERO_CONDITIONS_SHORT):
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            sr, se = muzero_solve_rate(arr, diff_dir, muzero_runs)
            names.append(muzero_short_labels.get(name, name))
            rates.append(sr * 100)
            ses.append(se * 100)
            colors.append(PLOT_COLORS[i % len(PLOT_COLORS)])

        for model_label, model_display in LLM_MODELS:
            model_colors = LLM_MODEL_COLORS[model_label]
            short_name = llm_short_names.get(model_label, model_display)
            for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
                file_stem = f"LLM_{model_label}_{strat}"
                data = load_llm_json(file_dir, file_stem)
                if data is None:
                    continue
                strat_display = "0-shot" if strat == "zero_shot" else strat
                names.append(f"{short_name}\n({strat_display})")
                rates.append(data["solve_rate"] * 100)
                ses.append(data.get("se_solve_rate", 0.0) * 100)
                colors.append(model_colors[j % len(model_colors)])

        ax = axs[col]
        _bar_group(ax, names, rates, ses, colors, diff_title,
                   ylabel="Solve rate (%)" if col == 0 else None,
                   value_fmt="{:.0f}%")
        ax.set_ylim(0, 115)

    fig.tight_layout()
    out = os.path.join(root_dir, f"SolveRate_Comparison_{timestamp}.pdf")
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
        ("MuZero", "MuZero", PLOT_COLORS[0]),
        ("Value ablated\n(PFC lesion)", "Value abl.\n(PFC)", PLOT_COLORS[2]),
        ("Policy ablated\n(Cerebellar)", "Policy abl.\n(Cereb.)", PLOT_COLORS[1]),
    ]
    llm_short_names = {"qwen25_7b": "Qwen", "llama3_8b": "Llama"}

    fig, axs = plt.subplots(1, len(DIFFICULTIES), figsize=(12, 3.5), sharey=True)

    for col, (diff_dir, diff_title) in enumerate(DIFFICULTIES):
        file_dir = os.path.join(root_dir, diff_dir)
        names, rates, ses, colors = [], [], [], []

        for cond_key, cond_display, cond_color in muzero_conds:
            rate, se = MUZERO_ILLEGAL_RATES_AGGREGATE[cond_key]
            names.append(cond_display)
            rates.append(rate)
            ses.append(se)
            colors.append(cond_color)

        for model_label, model_display in LLM_MODELS:
            model_colors = LLM_MODEL_COLORS[model_label]
            short_name = llm_short_names.get(model_label, model_display)
            for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
                file_stem = f"LLM_{model_label}_{strat}"
                data = load_llm_json(file_dir, file_stem)
                if data is None:
                    continue
                strat_display = "0-shot" if strat == "zero_shot" else strat
                names.append(f"{short_name}\n({strat_display})")
                rates.append(data["mean_illegal_rate"] * 100)
                ses.append(data["se_illegal_rate"] * 100)
                colors.append(model_colors[j % len(model_colors)])

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
    out = os.path.join(root_dir, f"IllegalRate_Comparison_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6b: Illegal move rate — condensed stacked bar (one bar per condition)
# ══════════════════════════════════════════════════════════════════════════════

def fig_illegal_rate_stacked(root_dir: str, timestamp: str):
    """Single-panel grouped bar chart: one group per difficulty, one bar per
    condition within each group. Conditions are distinguished by colour; the
    three difficulty groups form the x-axis.
    """
    def _darken(color, factor=0.75):
        r, g, b, *a = mpl.colors.to_rgba(color)
        return (r * factor, g * factor, b * factor, 1.0)

    font_s = 7
    mpl.rc("font", size=font_s)

    diff_labels  = {"LS": "Close\n(1 move)", "MS": "Moderate\n(3 moves)", "ES": "Far\n(7 moves)"}

    muzero_conds = [
        ("MuZero",                      "MuZero",             "#666666"),
        ("Value ablated\n(PFC lesion)",  "Value abl.\n(PFC)",  PLOT_COLORS[2]),
        ("Policy ablated\n(Cerebellar)", "Policy abl.\n(Cereb.)", PLOT_COLORS[1]),
    ]
    llm_short       = {"qwen25_7b": "Qwen", "llama3_8b": "Llama"}
    llm_strat_short = {"zero_shot": "0-shot", "cot": "CoT"}

    # Load per-difficulty MuZero rates
    per_diff_muzero = {}
    for diff_dir, _ in DIFFICULTIES:
        path = os.path.join(root_dir, diff_dir, "muzero_illegal_rates.json")
        if os.path.exists(path):
            with open(path) as f:
                per_diff_muzero[diff_dir] = json.load(f)

    # Build list of conditions: (display_name, color, {diff_dir: (rate, se)})
    conditions = []

    for cond_key, cond_display, cond_color in muzero_conds:
        diff_vals = {}
        for diff_dir, _ in DIFFICULTIES:
            if diff_dir in per_diff_muzero:
                d = per_diff_muzero[diff_dir]
                diff_vals[diff_dir] = (d[cond_key]["mean"] * 100, d[cond_key]["se"] * 100)
            else:
                rate, se = MUZERO_ILLEGAL_RATES_AGGREGATE[cond_key]
                diff_vals[diff_dir] = (rate, se)
        conditions.append((cond_display, cond_color, diff_vals))

    for model_label, _ in LLM_MODELS:
        model_colors = LLM_MODEL_COLORS[model_label]
        short = llm_short.get(model_label, model_label)
        for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
            strat_s = llm_strat_short.get(strat, strat)
            diff_vals = {}
            all_present = True
            for diff_dir, _ in DIFFICULTIES:
                data = load_llm_json(os.path.join(root_dir, diff_dir), f"LLM_{model_label}_{strat}")
                if data is None:
                    all_present = False
                    break
                diff_vals[diff_dir] = (data["mean_illegal_rate"] * 100, data["se_illegal_rate"] * 100)
            if all_present:
                conditions.append((f"{short}\n({strat_s})", model_colors[j % len(model_colors)], diff_vals))

    n_conds = len(conditions)
    n_diff  = len(DIFFICULTIES)
    bar_w    = 0.12
    group_sp = 1.0  # spacing between difficulty group centres
    offsets  = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_w
    x        = np.arange(n_diff) * group_sp

    fig, ax = plt.subplots(figsize=(max(4.0, n_diff * group_sp * 1.8), 3.5))

    legend_patches = []
    for ci, (cond_display, cond_color, diff_vals) in enumerate(conditions):
        rates  = np.array([diff_vals[diff_dir][0] for diff_dir, _ in DIFFICULTIES])
        ses    = np.array([diff_vals[diff_dir][1] for diff_dir, _ in DIFFICULTIES])
        dark_c = _darken(cond_color)

        ax.bar(
            x + offsets[ci], rates, bar_w,
            yerr=ses, capsize=2,
            color=cond_color,
            edgecolor=dark_c,
            linewidth=0.5,
            error_kw=dict(zorder=3, linewidth=0.8),
        )

        # Dot marker at y=0 so zero-rate bars are still visible
        zero_mask = rates == 0
        if zero_mask.any():
            ax.scatter(
                (x + offsets[ci])[zero_mask], np.full(zero_mask.sum(), 1.5),
                marker="o", s=12,
                color=cond_color,
                edgecolors=dark_c,
                linewidths=0.5,
                zorder=4,
            )

        legend_patches.append(
            mpl.patches.Patch(facecolor=cond_color, edgecolor=dark_c, label=cond_display)
        )

    ax.set_xticks(x)
    ax.set_xticklabels([diff_labels[diff_dir] for diff_dir, _ in DIFFICULTIES],
                       fontsize=font_s)
    ax.set_ylabel("Illegal move rate (%)")
    ax.set_ylim(0, 100)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.legend(handles=legend_patches, title="Model", fontsize=font_s - 1,
              title_fontsize=font_s - 1, loc="upper left", frameon=False,
              ncol=1)

    fig.tight_layout()
    out = os.path.join(root_dir, f"IllegalRate_Stacked_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6c: Illegal rate + mean error — line charts (shared legend)
# ══════════════════════════════════════════════════════════════════════════════

def fig_illegal_rate_and_error_lines(root_dir: str, timestamp: str,
                                     muzero_runs: int = 10,
                                     norm_by: str = "condition"):
    """Two-panel figure: illegal move rate (top) and mean error (bottom) as line
    charts across difficulty levels.  One line per condition; shared legend at
    the bottom.

    X-axis: difficulty (Close → Moderate → Far).
    """
    def _darken(color, factor=0.75):
        r, g, b, *a = mpl.colors.to_rgba(color)
        return (r * factor, g * factor, b * factor, 1.0)

    font_s = 7
    mpl.rc("font", size=font_s)

    se_factor = 1.0 / np.sqrt(muzero_runs)

    diff_order = [("LS", "Close\n(1 move)"), ("MS", "Moderate\n(3 moves)"), ("ES", "Far\n(7 moves)")]
    # Worst-case error per difficulty for normalisation (0 = optimal, 1 = fail)
    worst_case = {d: MAX_STEPS - OPTIMAL_MOVES[d] for d, _ in diff_order}

    muzero_conds = [
        ("MuZero",               "Muzero"),
        ("Value abl. (PFC)",     "ResetLatentVal"),
        ("Policy abl. (Cereb.)", "ResetLatentPol"),
    ]
    llm_short       = {"qwen25_7b": "Qwen", "llama3_8b": "Llama"}
    llm_strat_short = {"zero_shot": "0-shot", "cot": "CoT"}

    # One colour per model group; markers/linestyles distinguish conditions within group
    GROUP_COLORS = {
        "muzero":    "#2166AC",   # blue
        "qwen25_7b": "#D6604D",   # red-orange
        "llama3_8b": "#4DAC26",   # green
        "human":     "#756BB1",   # purple
    }
    GROUP_MARKERS = {
        "muzero":    ["o", "s", "^"],
        "qwen25_7b": ["o", "s"],
        "llama3_8b": ["o", "s"],
        "human":     ["o", "s", "^"],
    }
    GROUP_LINESTYLES = {
        "muzero":    ["-", "--", "-."],
        "qwen25_7b": ["-", "--"],
        "llama3_8b": ["-", "--"],
        "human":     ["--", "--", "--"],
    }
    group_idx_counter = {}   # track within-group index

    # Load per-difficulty MuZero illegal rates
    per_diff_muzero = {}
    for diff_dir, _ in diff_order:
        path = os.path.join(root_dir, diff_dir, "muzero_illegal_rates.json")
        if os.path.exists(path):
            with open(path) as f:
                per_diff_muzero[diff_dir] = json.load(f)

    # Build unified condition list: (display_name, color, group, illegal_vals, error_vals)
    # group is used for model-level normalisation ("muzero", model_label).
    # illegal_vals / error_vals: list of (mean, se) or None per difficulty.
    conditions = []

    MUZERO_KEY_MAP = {
        "Muzero":         "MuZero",
        "ResetLatentVal": "Value ablated\n(PFC lesion)",
        "ResetLatentPol": "Policy ablated\n(Cerebellar)",
    }

    # MuZero conditions
    for cond_display, file_label in muzero_conds:
        ill_vals, err_vals = [], []
        for diff_dir, _ in diff_order:
            if diff_dir in per_diff_muzero:
                d = per_diff_muzero[diff_dir]
                key = MUZERO_KEY_MAP[file_label]
                ill_vals.append((d[key]["mean"] * 100, d[key]["se"] * 100))
            else:
                rate, se = MUZERO_ILLEGAL_RATES_AGGREGATE[MUZERO_KEY_MAP[file_label]]
                ill_vals.append((rate, se))
            arr = load_accuracy(os.path.join(root_dir, diff_dir), file_label)
            if arr is not None:
                mean_e = float(arr[-1, 1])
                sd_e = float(arr[-1, 2]) if arr.shape[1] > 2 else 0.0
                err_vals.append((mean_e, sd_e * se_factor))
            else:
                err_vals.append(None)
        gi = group_idx_counter.get("muzero", 0)
        group_idx_counter["muzero"] = gi + 1
        conditions.append((cond_display, "muzero", gi, ill_vals, err_vals))

    # LLM conditions
    for model_label, _ in LLM_MODELS:
        short = llm_short.get(model_label, model_label)
        for j, strat in enumerate(LLM_PROMPTING_STRATEGIES):
            strat_s = llm_strat_short.get(strat, strat)
            ill_vals, err_vals = [], []
            any_data = False
            for diff_dir, _ in diff_order:
                data = load_llm_json(os.path.join(root_dir, diff_dir),
                                     f"LLM_{model_label}_{strat}")
                if data is not None:
                    ill_vals.append((data["mean_illegal_rate"] * 100, data["se_illegal_rate"] * 100))
                    err_vals.append((data["mean_error"], data["se_error"]))
                    any_data = True
                else:
                    ill_vals.append(None)
                    err_vals.append(None)
            if any_data:
                gi = group_idx_counter.get(model_label, 0)
                group_idx_counter[model_label] = gi + 1
                conditions.append((f"{short} ({strat_s})", model_label, gi, ill_vals, err_vals))

    # ── Human reference data from CSV ─────────────────────────────────────────
    # Optimal moves per difficulty for Goel et al. 2001 (5-disk ToH)
    _goel_optimal = {"LS": 7.0, "MS": 10.6, "ES": 14.3}

    csv_path = os.path.join(root_dir, "tables", "planning_comparison_human_only.csv")
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path) as _f:
            rows = list(_csv.DictReader(_f))

        def _get_row(source_substr, metric):
            for r in rows:
                if metric in r["metric"] and source_substr.lower() in r["source"].lower():
                    return r
            return None

        def _diff_vals(row, keys=("difficulty_close_easy", "difficulty_moderate_medium", "difficulty_far_hard")):
            out = []
            for k in keys:
                v = row[k].strip()
                out.append((float(v), 0.0) if v not in ("NA", "", "nan") else None)
            return out

        # NC controls — illegal rate (assumed baseline, Grafman 1992)
        r_nc_ill = _get_row("NC controls — Grafman", "illegal_move_rate")
        # CCA patients — illegal rate (schematic, Grafman 1992)
        r_cca_ill = _get_row("CCA patients", "illegal_move_rate")
        # NC controls — moves by difficulty (Goel et al. 2001)
        r_nc_err  = _get_row("NC controls solved-only — Goel", "moves_per_problem_by_difficulty")
        r_nc_sr   = _get_row("NC controls — Goel", "solve_rate")
        # PFC patients — moves by difficulty (Goel et al. 2001)
        r_pfc_err = _get_row("PFC patients solved-only", "moves_per_problem_by_difficulty")
        r_pfc_sr  = _get_row("PFC patients — Goel", "solve_rate")

        def _moves_to_error_imputed(moves_row, sr_row):
            """Impute failed trials as 2× optimal moves, weighted by solve rate.

            imputed_mean = solve_rate * solved_moves + (1 - solve_rate) * (2 * optimal)
            imputed_error = imputed_mean - optimal
            """
            keys = ("difficulty_close_easy", "difficulty_moderate_medium", "difficulty_far_hard")
            diffs = [d for d, _ in diff_order]
            out = []
            for k, d in zip(keys, diffs):
                mv = moves_row[k].strip() if moves_row else "NA"
                sr = sr_row[k].strip()   if sr_row   else "NA"
                if mv not in ("NA", "", "nan") and sr not in ("NA", "", "nan"):
                    solve_rate   = float(sr) / 100.0
                    solved_moves = float(mv)
                    opt          = _goel_optimal[d]
                    fail_moves   = 2.0 * opt
                    imputed_mean = solve_rate * solved_moves + (1 - solve_rate) * fail_moves
                    out.append((imputed_mean - opt, 0.0))
                elif mv not in ("NA", "", "nan"):
                    # No solve rate — fall back to solved-only (original behaviour)
                    out.append((float(mv) - _goel_optimal[d], 0.0))
                else:
                    out.append(None)
            return out

        # Build human conditions:
        # NC controls — ill from Grafman, err from Goel (same population label)
        nc_ill = _diff_vals(r_nc_ill) if r_nc_ill else [None] * 3
        gi = group_idx_counter.get("human", 0); group_idx_counter["human"] = gi + 1
        conditions.append(("NC controls (reference)", "human", gi, nc_ill, [None] * 3))

        # CCA patients — ill only (Grafman schematic), no error data
        cca_ill = _diff_vals(r_cca_ill) if r_cca_ill else [None] * 3
        gi = group_idx_counter.get("human", 0); group_idx_counter["human"] = gi + 1
        conditions.append(("CA patients (schematic)", "human", gi, cca_ill, [None] * 3))

        # PFC patients — illegal rate only; error data removed from this panel
        gi = group_idx_counter.get("human", 0); group_idx_counter["human"] = gi + 1
        conditions.append(("PFC patients (reference)", "human", gi, [None] * 3, [None] * 3))

    # ── Normalisation ─────────────────────────────────────────────────────────
    def _norm_vals(vals, lo, hi):
        rng = hi - lo if hi != lo else 1.0
        return [((v[0] - lo) / rng, None) if v is not None else None for v in vals]

    if norm_by == "model":
        # Compute per-group (lo, hi) across all conditions and difficulties
        from collections import defaultdict
        group_ill_means = defaultdict(list)
        group_err_means = defaultdict(list)
        for _, grp, gi, iv, ev in conditions:
            group_ill_means[grp].extend(v[0] for v in iv if v is not None)
            group_err_means[grp].extend(v[0] for v in ev if v is not None)
        group_ill_range = {g: (min(vs), max(vs)) for g, vs in group_ill_means.items() if vs}
        group_err_range = {g: (min(vs), max(vs)) for g, vs in group_err_means.items() if vs}

        normed = []
        for name, grp, gi, iv, ev in conditions:
            lo_i, hi_i = group_ill_range.get(grp, (0, 1))
            lo_e, hi_e = group_err_range.get(grp, (0, 1))
            normed.append((name, grp, gi, _norm_vals(iv, lo_i, hi_i),
                           _norm_vals(ev, lo_e, hi_e)))
    else:
        # Per-condition normalisation (original behaviour)
        normed = []
        for name, grp, gi, iv, ev in conditions:
            ill_means = [v[0] for v in iv if v is not None]
            err_means = [v[0] for v in ev if v is not None]
            lo_i, hi_i = (min(ill_means), max(ill_means)) if ill_means else (0, 1)
            lo_e, hi_e = (min(err_means), max(err_means)) if err_means else (0, 1)
            normed.append((name, grp, gi, _norm_vals(iv, lo_i, hi_i),
                           _norm_vals(ev, lo_e, hi_e)))

    x = np.arange(len(diff_order))
    x_labels = [label for _, label in diff_order]

    fig, (ax_sr, ax_err, ax_ill) = plt.subplots(3, 1, figsize=(5.5, 5.0), sharex=True)

    # Per-(condition, x-position) dodge: only shift conditions whose y-value
    # is within overlap_thr of another condition at the same x position.
    def _compute_dodges(normed, n_x, overlap_thr=0.06, dodge_amt=0.02):
        n_conds = len(normed)
        dx = [[0.0] * n_x for _ in range(n_conds)]
        for xi in range(n_x):
            ys = {}
            for ci, (_, _, _, norm_ill, norm_err) in enumerate(normed):
                v = norm_ill[xi] if (xi < len(norm_ill) and norm_ill[xi] is not None) else \
                    (norm_err[xi] if (xi < len(norm_err) and norm_err[xi] is not None) else None)
                if v is not None:
                    ys[ci] = v[0]
            if not ys:
                continue
            sorted_conds = sorted(ys.items(), key=lambda t: t[1])
            groups, cur = [], [sorted_conds[0]]
            for i in range(1, len(sorted_conds)):
                if abs(sorted_conds[i][1] - sorted_conds[i - 1][1]) < overlap_thr:
                    cur.append(sorted_conds[i])
                else:
                    groups.append(cur)
                    cur = [sorted_conds[i]]
            groups.append(cur)
            for group in groups:
                if len(group) > 1:
                    offsets = np.linspace(-(len(group) - 1) / 2,
                                          (len(group) - 1) / 2,
                                          len(group)) * dodge_amt
                    for (ci, _), off in zip(group, offsets):
                        dx[ci][xi] = off
        return dx

    dodge_dx = _compute_dodges(normed, len(diff_order))

    human_names = {"NC controls (reference)", "CA patients (schematic)", "PFC patients (reference)"}

    legend_handles = []
    added_to_legend = set()
    for ci, (cond_display, grp, gi, norm_ill, norm_err) in enumerate(normed):
        is_human  = cond_display in human_names
        color     = GROUP_COLORS.get(grp, "#888888")
        dark_c    = _darken(color)
        mlist     = GROUP_MARKERS.get(grp, ["o", "s", "^", "D"])
        lslist    = GROUP_LINESTYLES.get(grp, ["-", "--", "-.", ":"])
        marker    = mlist[gi % len(mlist)]
        ls        = lslist[gi % len(lslist)]
        alpha     = 0.7 if is_human else 1.0
        lw        = 1.5 if is_human else 2.0

        # ── Top panel: normalised illegal rate ────────────────────────────────
        ill_x, ill_y = [], []
        for xi, v in enumerate(norm_ill):
            if v is not None:
                ill_x.append(x[xi] + dodge_dx[ci][xi])
                ill_y.append(v[0])
        if ill_x:
            ax_ill.plot(ill_x, ill_y, color=color, marker=marker,
                        markersize=6 if is_human else 5,
                        linestyle=ls, linewidth=lw, alpha=alpha,
                        markeredgecolor=dark_c, markeredgewidth=0.6)

        # ── Bottom panel: normalised mean error ───────────────────────────────
        err_x, err_y = [], []
        for xi, v in enumerate(norm_err):
            if v is not None:
                err_x.append(x[xi] + dodge_dx[ci][xi])
                err_y.append(v[0])
        if err_x:
            ax_err.plot(err_x, err_y, color=color, marker=marker,
                        markersize=6 if is_human else 5,
                        linestyle=ls, linewidth=lw, alpha=alpha,
                        markeredgecolor=dark_c, markeredgewidth=0.6)

        # Add to legend once
        if cond_display not in added_to_legend and (ill_x or err_x):
            legend_handles.append(
                mpl.lines.Line2D([0], [0], color=color, marker=marker,
                                 markersize=6 if is_human else 5,
                                 linewidth=lw, linestyle=ls, alpha=alpha,
                                 markeredgecolor=dark_c, markeredgewidth=0.6,
                                 label=cond_display)
            )
            added_to_legend.add(cond_display)

    # ── Top panel: human solve rate (absolute) ───────────────────────────────
    sr_legend_handles = []
    if os.path.exists(csv_path):
        sr_human = [
            ("NC controls",  _get_row("NC controls — Goel", "solve_rate"),  "o", "-"),
            ("PFC patients", _get_row("PFC patients — Goel", "solve_rate"), "s", "--"),
        ]
        human_color = GROUP_COLORS["human"]
        human_dark  = _darken(human_color)
        for label, row, marker, ls in sr_human:
            if row is None:
                continue
            sr_vals = _diff_vals(row)
            sr_x = [x[xi]         for xi, v in enumerate(sr_vals) if v is not None]
            sr_y = [100.0 - v[0]  for v      in sr_vals             if v is not None]
            if sr_x:
                ax_sr.plot(sr_x, sr_y, color=human_color, marker=marker,
                           markersize=5, linestyle=ls, linewidth=2.0,
                           markeredgecolor=human_dark, markeredgewidth=0.6)
                sr_legend_handles.append(
                    mpl.lines.Line2D([0], [0], color=human_color, marker=marker,
                                     markersize=5, linewidth=2.0, linestyle=ls,
                                     markeredgecolor=human_dark, markeredgewidth=0.6,
                                     label=label)
                )

    ax_sr.set_ylabel("Unsolved rate (%)", fontsize=font_s)
    ax_sr.set_ylim(-5, 105)
    ax_sr.spines["right"].set_visible(False)
    ax_sr.spines["top"].set_visible(False)
    ax_sr.tick_params(which="both", left=True, bottom=True,
                      top=False, right=False, direction="out", labelsize=font_s)

    ax_err.set_ylabel("Norm. mean error", fontsize=font_s)
    ax_err.set_ylim(-0.05, 1.05)
    ax_err.spines["right"].set_visible(False)
    ax_err.spines["top"].set_visible(False)
    ax_err.tick_params(which="both", left=True, bottom=True,
                       top=False, right=False, direction="out", labelsize=font_s)

    ax_ill.set_ylabel("Norm. illegal move rate", fontsize=font_s)
    ax_ill.set_ylim(-0.05, 1.05)
    ax_ill.set_xticks(x)
    ax_ill.set_xticklabels(x_labels, fontsize=font_s)
    ax_ill.spines["right"].set_visible(False)
    ax_ill.spines["top"].set_visible(False)
    ax_ill.tick_params(which="both", left=True, bottom=True,
                       top=False, right=False, direction="out", labelsize=font_s)

    fig.legend(handles=legend_handles, title="Model", fontsize=font_s - 1,
               title_fontsize=font_s - 1, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               ncol=min(len(legend_handles), 4))

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    suffix = "_model_norm" if norm_by == "model" else ""
    out = os.path.join(root_dir, f"IllegalRate_MeanError_Lines{suffix}_{timestamp}.pdf")
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
    out = os.path.join(root_dir, f"LLM_FeedbackSweep_{timestamp}.pdf")
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
    out = os.path.join(root_dir, f"LLM_LayerSweep_AllDifficulties_{timestamp}.pdf")
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
        out_s = os.path.join(diff_root, f"LLM_LayerSweep_{timestamp}.pdf")
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

    out = os.path.join(root_dir, f"LLM_AblationMatrix_{timestamp}.pdf")
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
    out = os.path.join(root_dir, f"MuZero_AblationMatrix_{timestamp}.pdf")
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
    out = os.path.join(root_dir, f"LLM_ConditionMatrix_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Two-model comparison heatmap (Qwen vs Llama)
# ══════════════════════════════════════════════════════════════════════════════

CROSS_MODEL_CONDITIONS = [
    ("zero_shot", "Zero-shot"),
    ("cot", "CoT"),
    ("cot_h5_illfb", "CoT + horizon + illegal fb"),
    ("cot_ablateL0", "CoT + ablate layer 0"),
    ("cot_ablateL14", "CoT + ablate layer 14"),
]


def fig_cross_model_heatmap(root_dir: str, timestamp: str):
    """Side-by-side heatmaps comparing Qwen and Llama on shared conditions."""
    font_s = 7
    mpl.rc("font", size=font_s)

    col_labels = [name for _, name in HEATMAP_DIFFICULTIES]
    row_labels = [name for _, name in CROSS_MODEL_CONDITIONS]

    n_rows = len(CROSS_MODEL_CONDITIONS)
    n_cols = len(HEATMAP_DIFFICULTIES)

    fig, (ax_q, ax_l) = plt.subplots(2, 1, figsize=(4.5, 5.5),
                                      gridspec_kw={"hspace": 0.35})

    for ax, (model_label, model_display) in zip([ax_q, ax_l], LLM_MODELS):
        data = np.full((n_rows, n_cols), np.nan)
        se_data = np.full((n_rows, n_cols), np.nan)

        for j, (diff_dir, _) in enumerate(HEATMAP_DIFFICULTIES):
            file_dir = os.path.join(root_dir, diff_dir)
            for i, (stem, _) in enumerate(CROSS_MODEL_CONDITIONS):
                res = load_llm_json(file_dir, f"LLM_{model_label}_{stem}")
                if res is not None:
                    data[i, j] = res["mean_error"]
                    se_data[i, j] = res["se_error"]

        _draw_heatmap(ax, data, se_data, row_labels, col_labels,
                      title=model_display,
                      cbar_label="Mean error",
                      vmin=0, vmax=200)
        ax.set_ylabel("Condition", fontsize=8)

    # Shared colorbar
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=60, vmax=200)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_q, ax_l], shrink=0.8, pad=0.02)
    cbar.set_label("Mean error (steps above optimal)", fontsize=7)

    fig.tight_layout()
    out = os.path.join(root_dir, f"LLM_CrossModel_Comparison_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


CROSS_MODEL_CONDITIONS_COMBINED = [
    ("zero_shot", "Zero-shot"),
    ("cot", "CoT"),
    ("cot_h5_illfb", "CoT + horizon + illegal fb"),
]


def fig_cross_model_heatmap_combined(root_dir: str, timestamp: str,
                                     out_dir: str = None):
    """Single combined heatmap with model name in the row labels."""
    if out_dir is None:
        out_dir = root_dir

    font_s = 7
    mpl.rc("font", size=font_s)

    col_labels = [name for _, name in HEATMAP_DIFFICULTIES]

    n_conds = len(CROSS_MODEL_CONDITIONS_COMBINED)
    n_cols = len(HEATMAP_DIFFICULTIES)
    n_models = len(LLM_MODELS)
    n_rows = n_models * n_conds

    data = np.full((n_rows, n_cols), np.nan)
    se_data = np.full((n_rows, n_cols), np.nan)
    row_labels = []

    for m, (model_label, model_display) in enumerate(LLM_MODELS):
        for i, (stem, cond_name) in enumerate(CROSS_MODEL_CONDITIONS_COMBINED):
            row_idx = m * n_conds + i
            row_labels.append(f"{model_display} — {cond_name}")
            for j, (diff_dir, _) in enumerate(HEATMAP_DIFFICULTIES):
                file_dir = os.path.join(root_dir, diff_dir)
                res = load_llm_json(file_dir, f"LLM_{model_label}_{stem}")
                if res is not None:
                    data[row_idx, j] = res["mean_error"]
                    se_data[row_idx, j] = res["se_error"]

    row_height = 0.38
    fig, ax = plt.subplots(figsize=(5.5, row_height * n_rows + 1.5))

    _draw_heatmap(ax, data, se_data, row_labels, col_labels,
                  title="Cross-Model Comparison",
                  cbar_label="Mean error (steps above optimal)",
                  vmin=0, vmax=200)

    # Draw a horizontal separator between models
    ax.axhline(y=n_conds - 0.5, color="white", linewidth=2.5)

    ax.set_ylabel("", fontsize=8)

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=60, vmax=200)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean error (steps above optimal)", fontsize=7)

    fig.tight_layout()
    out = os.path.join(out_dir, f"LLM_CrossModel_Combined_{timestamp}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Parse failures vs error scatter
# ══════════════════════════════════════════════════════════════════════════════

PARSE_MODELS = {
    "qwen25_7b": {"display": "Qwen-2.5 7B", "marker": "o"},
    "llama3_8b": {"display": "Llama-3 8B", "marker": "D"},
}
PARSE_DIFF_COLORS = {"LS": "#4C72B0", "MS": "#DD8452", "ES": "#55A868"}
PARSE_CONDITION_LABELS = {
    "cot_ablateL0": "ablate L0",
    "cot_ablateL0_h5_illfb": "ablate L0 +fb",
    "cot_noiseS0.5_L0": "noise L0",
    "cot_noiseS0.5_L0_h5_illfb": "noise L0 +fb",
    "cot_ablateL14": "ablate L14",
    "cot_noiseS0.5_L8": "noise L8",
}


def _parse_stem(stem):
    rest = stem.replace("LLM_", "")
    for model_key in PARSE_MODELS:
        if rest.startswith(model_key + "_"):
            return model_key, rest[len(model_key) + 1:]
    return None, rest


def fig_parse_failures(root_dir: str, timestamp: str):
    """Scatter: mean parse failures per episode vs mean error, all LLM conditions."""
    rows = []
    for diff_dir, diff_name in DIFFICULTIES:
        for path in sorted(glob.glob(os.path.join(root_dir, diff_dir, "LLM_*_results.json"))):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                continue  # skip empty / malformed files
            episodes = data.get("episodes", [])
            if not episodes:
                continue
            mean_pf = np.mean([ep["parse_failures"] for ep in episodes])
            stem = os.path.basename(path).replace("_results.json", "")
            model_key, condition = _parse_stem(stem)
            rows.append({
                "model": model_key, "condition": condition,
                "difficulty": diff_dir, "diff_name": diff_name,
                "mean_error": data["mean_error"],
                "mean_parse_failures": mean_pf,
            })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for diff_dir, diff_name in DIFFICULTIES:
        for model_key, minfo in PARSE_MODELS.items():
            subset = [r for r in rows
                      if r["difficulty"] == diff_dir and r["model"] == model_key]
            if not subset:
                continue
            ax.scatter(
                [r["mean_parse_failures"] for r in subset],
                [r["mean_error"] for r in subset],
                marker=minfo["marker"],
                color=PARSE_DIFF_COLORS[diff_dir],
                s=35, alpha=0.75,
                edgecolors="black", linewidths=0.4,
            )

    # Annotate outliers (parse_failures > 1)
    outliers = sorted(
        [r for r in rows if r["mean_parse_failures"] > 1],
        key=lambda r: (r["mean_parse_failures"], r["mean_error"]),
    )
    used_positions = []
    for r in outliers:
        model_short = PARSE_MODELS[r["model"]]["display"].split()[0]
        cond_label = PARSE_CONDITION_LABELS.get(r["condition"], r["condition"])
        label = f"{model_short}, {cond_label} ({r['diff_name']})"
        base_x, base_y = 14, 0
        for dy_idx in range(len(used_positions) + 1):
            dy = -12 * dy_idx
            candidate = (r["mean_parse_failures"] + base_x,
                         r["mean_error"] + base_y + dy)
            if not any(abs(candidate[0] - px) < 20 and abs(candidate[1] - py) < 10
                       for px, py in used_positions):
                break
        used_positions.append(candidate)
        ax.annotate(
            label, (r["mean_parse_failures"], r["mean_error"]),
            fontsize=5.5, alpha=0.85,
            xytext=(base_x, base_y + dy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
        )

    # Two-part legend: colour = difficulty, marker = model
    from matplotlib.lines import Line2D
    handles = []
    for diff_dir, diff_name in DIFFICULTIES:
        handles.append(Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=PARSE_DIFF_COLORS[diff_dir],
            markeredgecolor="black", markersize=7, linewidth=0, label=diff_name,
        ))
    handles.append(Line2D([0], [0], linestyle="", label=""))
    for model_key, minfo in PARSE_MODELS.items():
        handles.append(Line2D(
            [0], [0], marker=minfo["marker"], color="w", markerfacecolor="grey",
            markeredgecolor="black", markersize=7, linewidth=0, label=minfo["display"],
        ))
    ax.set_xlabel("Mean parse failures per episode")
    ax.set_ylabel("Mean error (excess moves)")
    ax.set_title("Parse failures vs. error — all LLM conditions")
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="center right")

    fig.tight_layout()
    out = os.path.join(root_dir, f"ParseFailures_vs_Error_{timestamp}.pdf")
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
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                continue  # skip empty / malformed files
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
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                continue  # skip empty / malformed files
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

    print("[1/13] MuZero ablation grid")
    fig_muzero_ablation_grid(root_dir, args.timestamp, args.muzero_runs)

    print("[2/13] MuZero bar charts (simulations to baseline)")
    fig_muzero_bar_charts(root_dir, args.timestamp, args.muzero_runs)

    print("[3/13] MuZero average performance")
    fig_muzero_average_performance(root_dir, args.timestamp, args.muzero_runs)

    print("[4/13] LLM vs MuZero — error comparison")
    fig_llm_muzero_error(root_dir, args.timestamp, args.muzero_runs)

    print("[5/13] Solve rate comparison")
    fig_solve_rate(root_dir, args.timestamp, args.muzero_runs)

    print("[6/13] Illegal move rate comparison")
    fig_illegal_rate(root_dir, args.timestamp)

    print("[6b/13] Illegal move rate — condensed stacked")
    fig_illegal_rate_stacked(root_dir, args.timestamp)

    print("[6c/13] Illegal rate + mean error — line charts (condition-normalised)")
    fig_illegal_rate_and_error_lines(root_dir, args.timestamp, args.muzero_runs, norm_by="condition")

    print("[6d/13] Illegal rate + mean error — line charts (model-normalised)")
    fig_illegal_rate_and_error_lines(root_dir, args.timestamp, args.muzero_runs, norm_by="model")

    print("[7/13] LLM feedback sweep")
    fig_feedback_sweep(root_dir, args.timestamp)

    print("[8/13] LLM layer ablation")
    fig_layer_ablation(root_dir, args.timestamp)

    print("[9/13] LLM ablation matrix")
    fig_llm_ablation_matrix(root_dir, args.timestamp)

    print("[10/13] MuZero heatmap")
    fig_muzero_heatmap(root_dir, args.timestamp, args.muzero_runs)

    print("[11/13] LLM heatmap")
    fig_llm_heatmap(root_dir, args.timestamp)

    print("[12/13] Cross-model comparison heatmap")
    fig_cross_model_heatmap(root_dir, args.timestamp)

    print("[13/13] Parse failures vs error")
    fig_parse_failures(root_dir, args.timestamp)

    print("\n[Tables] Generating CSV summaries")
    generate_tables(root_dir, args.muzero_runs)

    print("\nDone! All figures and tables regenerated.")


if __name__ == "__main__":
    main()
