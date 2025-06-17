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

label_1 = "Muzero"
label_2 = "ResetLatentPol"
label_3 = "ResetLatentVal"
lable_4 = "ResetLatentRwd"
label_5 = "ResetLatentVal_ResetLatentRwd"

labels = [label_1, label_2, label_3, lable_4, label_5]

set_plot_style()

font_s = 7
mpl.rc("font", size=font_s)
mpl.rcParams["xtick.labelsize"] = font_s
mpl.rcParams["ytick.labelsize"] = font_s

col_colors = PLOT_COLORS[: len(labels)]

fig, axs = plt.subplots(
    nrows=len(directories),
    ncols=len(labels),
    figsize=(7.5, 4),
    gridspec_kw={
        "wspace": 0.32,
        "hspace": 0.3,
        "left": 0.065,
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
        axs[e, i].plot(r[:, 0], r[:, 1], color=col_colors[i])
        axs[e, i].set_ylim([0, 100])
        axs[e, i].spines["right"].set_visible(False)
        axs[e, i].spines["top"].set_visible(False)
        if i == 0:
            if e == 0:
                axs[e, i].set_ylabel("Far\nError", fontsize=font_s)
            elif e == 1:
                axs[e, i].set_ylabel("Mid\nError", fontsize=font_s)
            elif e == 2:
                axs[e, i].set_ylabel("Close\nError", fontsize=font_s)
        if e == 0:
            axs[e, i].set_title(labels[i], fontsize=font_s)
        if e == len(directories) - 1:
            axs[e, i].set_xlabel(
                "N. simulations every real step \n (planning time)",
                fontsize=font_s,
            )
        i += 1
    e += 1

fig_bar, axs_bar = plt.subplots(
    nrows=1, ncols=len(directories), figsize=(7.5, 2.2), sharey=True
)

for e, d in enumerate(directories):
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

    bars = axs_bar[e].bar(labels, times_to_reach, color=col_colors, edgecolor="black")
    axs_bar[e].set_title(["Far", "Mid", "Close"][e])
    axs_bar[e].set_ylabel("Simulations to\nreach MuZero \n mean error")
    axs_bar[e].tick_params(axis="x", rotation=45)

    # Add hatching and/or asterisks for "never reached"
    for idx, (bar, never) in enumerate(zip(bars, never_reached_mask)):
        if never:
            bar.set_hatch("//")
            # Optional: Add asterisk above the bar
            axs_bar[e].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,  # adjust as needed
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
    nrows=1, ncols=len(directories), figsize=(7.5, 2.2), sharey=True
)

for e, d in enumerate(directories):
    results = []
    file_dir = os.path.join(root_dir, d)
    for l in labels:
        # Load and get acting accuracy column
        arr = torch.load(os.path.join(file_dir, l + "_actingAccuracy.pt")).numpy()
        results.append(arr[:, 1].mean())  # Take mean acting accuracy
    # Plot as bar chart
    bars = axs_avg[e].bar(labels, results, color=col_colors, edgecolor="black")
    axs_avg[e].set_title(["Far", "Mid", "Close"][e])
    axs_avg[e].set_ylabel("Mean acting accuracy\n(lower is better)")
    axs_avg[e].tick_params(axis="x", rotation=45)
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
