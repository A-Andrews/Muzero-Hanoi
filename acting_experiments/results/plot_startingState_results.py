import argparse
import os
import sys

sys.path.append("/well/costa/users/zqa082/Muzero-Hanoi")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import PLOT_COLORS, set_plot_style

parser = argparse.ArgumentParser(description="Plotting Results")
parser.add_argument(
    "--timestamp", type=str, default=None, help="Timestamp for the results directory"
)

args = parser.parse_args()
timestamp = args.timestamp

## Plot performance for tower of hanoi starting from pre-determined states at a fixed distance from the target
# e.g., ES: far from goal, MS: moderate, LS: close to goal.
## Load data

# Create directory to store results
root_dir = os.path.join("stats", "Hanoi", timestamp)

dir_1 = "ES"
dir_2 = "MS"
dir_3 = "LS"

directories = [dir_1, dir_2, dir_3]

directories_bar = [dir_3, dir_2, dir_1]
state_titles = ["Close to goal", "Mid distance", "Far from goal"]

label_1 = "Muzero"
label_2 = "ResetLatentPol"
label_3 = "ResetLatentVal"
lable_4 = "ResetLatentRwd"
label_5 = "ResetLatentVal_ResetLatentRwd"

labels = [label_1, label_2, label_3, lable_4, label_5]

name_1 = "Muzero"
name_2 = "Policy Ablated"
name_3 = "Value Ablated"
name_4 = "Reward Ablated"
name_5 = "Value + Reward Ablated"

names = [name_1, name_2, name_3, name_4, name_5]

set_plot_style()


def load_accuracy(directory: str, label: str) -> np.ndarray:
    """Load saved accuracy array with optional variance column."""
    path = os.path.join(directory, label + "_actingAccuracy.pt")
    arr = torch.load(path)
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    return np.asarray(arr, dtype=float)


font_s = 7
mpl.rc("font", size=font_s)
mpl.rcParams["xtick.labelsize"] = font_s
mpl.rcParams["ytick.labelsize"] = font_s

col_colors = PLOT_COLORS

fig, axs = plt.subplots(
    nrows=len(directories),
    ncols=len(labels),
    figsize=(7.5, 4),
    gridspec_kw={
        "wspace": 0.32,
        "hspace": 0.3,
        "left": 0.1,
        "right": 0.97,
        "bottom": 0.15,
        "top": 0.95,
    },
)

# Iterate through directories to plot each row with different ablations
e = 0
for d in directories:
    results = []
    file_dir = os.path.join(root_dir, d)
    print(file_dir)
    for l in labels:
        results.append(
            torch.load(os.path.join(file_dir, l + "_actingAccuracy.pt")).numpy()
        )
    print([r.shape for r in results])
    results = np.array(results)

    i = 0
    for r in results:
        errs = np.zeros_like(r[:, 1])
        axs[e, i].errorbar(
            r[:, 0],
            r[:, 1],
            yerr=errs,
            fmt="-o",
            color=col_colors[i % len(col_colors)],
            markersize=4,
            capsize=3,
        )
        axs[e, i].set_ylim([0, 100])
        axs[e, i].spines["right"].set_visible(False)
        axs[e, i].spines["top"].set_visible(False)
        if i != 0:
            axs[e, i].tick_params(axis="y", left=False, labelleft=False)
        if i == 0:
            if e == 0:
                axs[e, i].set_ylabel("Far\nError", fontsize=font_s)
            elif e == 1:
                axs[e, i].set_ylabel("Mid\nError", fontsize=font_s)
            elif e == 2:
                axs[e, i].set_ylabel("Close\nError", fontsize=font_s)
        if e == 0:
            axs[e, i].set_title(names[i], fontsize=font_s)
        if e == len(directories) - 1:
            axs[e, i].set_xlabel(
                "N. simulations per step \n (planning time)",
                fontsize=font_s,
            )
        i += 1
    e += 1

fig.tight_layout()
fig.savefig(
    os.path.join(root_dir, f"MuZero_Ablation_Comparison_{timestamp}.png"),
    dpi=1200,
)

labels = [label_1, label_2, label_3]
names = [name_1, name_2, name_3]

fig_bar, axs_bar = plt.subplots(
    nrows=1, ncols=len(directories_bar), figsize=(7.5, 3), sharey=True
)

for e, d in enumerate(directories_bar):
    results = []
    file_dir = os.path.join(root_dir, d)
    for l in labels:
        results.append(
            torch.load(os.path.join(file_dir, l + "_actingAccuracy.pt")).numpy()
        )
    results = np.array(results)
    mu_zero_avg = results[0][:, 1].mean()

    times_to_reach = []
    never_reached_mask = []
    for j, r in enumerate(results):
        indices = np.where(r[:, 1] <= mu_zero_avg)[0]
        if len(indices) > 0:
            times_to_reach.append(r[indices[0], 0])
            never_reached_mask.append(False)
        else:
            times_to_reach.append(r[-1, 0])
            never_reached_mask.append(True)

    bars = axs_bar[e].bar(
        names,
        times_to_reach,
        yerr=np.zeros_like(times_to_reach),
        capsize=5,
        color=[col_colors[i % len(col_colors)] for i in range(len(names))],
        edgecolor="none",
    )
    max_height = max(times_to_reach)
    label_offset = max_height * 0.05
    axs_bar[e].set_ylim(0, max_height * 1.25)

    axs_bar[e].set_title(state_titles[e], pad=10)
    if e == 0:
        axs_bar[e].set_ylabel("Simulations to\nbase rate")
    axs_bar[e].spines["right"].set_visible(False)
    axs_bar[e].spines["top"].set_visible(False)
    if e == len(directories_bar) - 1:
        axs_bar[e].legend(
            bars, names, fontsize=font_s, bbox_to_anchor=(1.05, 1), loc="upper right"
        )
    axs_bar[e].get_xaxis().set_visible(False)

    # Add hatching and/or asterisks for "never reached"
    for idx, (bar, never) in enumerate(zip(bars, never_reached_mask)):
        if never:
            bar.set_hatch("//")
            # Optional: Add asterisk above the bar
            axs_bar[e].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,  # adjust as needed
                "*",
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
            )
        else:
            # If you want, annotate the value for normal bars as well
            axs_bar[e].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=9,
            )

fig_bar.tight_layout()
fig_bar.savefig(
    os.path.join(root_dir, f"MuZero_Ablation_BarCharts_{timestamp}.png"), dpi=1200
)

# === Add Average Performance Bar Chart ===

fig_avg, axs_avg = plt.subplots(
    nrows=1, ncols=len(directories_bar), figsize=(7.5, 3), sharey=True
)

for e, d in enumerate(directories_bar):
    results = []
    file_dir = os.path.join(root_dir, d)
    for l in labels:
        # Load and get acting accuracy column
        arr = torch.load(os.path.join(file_dir, l + "_actingAccuracy.pt")).numpy()
        results.append(arr[:, 1].mean())  # Take mean acting accuracy
    # Plot as bar chart
    bars = axs_avg[e].bar(
        names,
        results,
        yerr=np.zeros_like(results),
        capsize=5,
        color=[col_colors[i % len(col_colors)] for i in range(len(names))],
        edgecolor="none",
    )
    axs_avg[e].get_xaxis().set_visible(False)
    axs_avg[e].set_title(state_titles[e])
    if e == 0:
        axs_avg[e].set_ylabel("Mean Error")
    axs_avg[e].spines["right"].set_visible(False)
    axs_avg[e].spines["top"].set_visible(False)

    if e == len(directories_bar) - 1:
        axs_avg[e].legend(
            bars, names, fontsize=font_s, bbox_to_anchor=(1.05, 1), loc="upper right"
        )

    # Annotate bars with their value
    for idx, bar in enumerate(bars):
        axs_avg[e].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=9,
        )

fig_avg.tight_layout()
fig_avg.savefig(
    os.path.join(root_dir, f"MuZero_Ablation_AveragePerformance_{timestamp}.png"),
    dpi=1200,
)
