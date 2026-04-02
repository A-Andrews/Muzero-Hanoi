"""
Schematic figure of the 3-disk Tower of Hanoi task (Start → Goal).
Pegs are labelled 0, 1, 2 (left to right).
Outputs: figures/hanoi_schematic.pdf
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils import set_plot_style

set_plot_style()

# --- Layout constants ---
N_DISKS = 3
PEG_X = [0.25, 0.5, 0.75]  # x positions of pegs; inset so widest disk fits
BASE_Y = 0.05               # y of the base bar
DISK_H = 0.10               # height of each disk in data coords (equal aspect)
DISK_FILL = 0.72            # fraction of DISK_H that is solid (rest is gap)
# peg ends just above the top disk
PEG_H = (N_DISKS - 1) * DISK_H + DISK_H * DISK_FILL + 0.02
# widths: index 0 = disk 1 (smallest), index 2 = disk 3 (largest)
# with equal aspect largest disk is 0.32 / (0.10*0.72) ≈ 4.4:1 (flat)
DISK_W = [0.16, 0.24, 0.32]
DISK_COLORS = ["#7BAFD4", "#4A7FA5", "#1F4E79"]   # light → dark blue (small → large)


def draw_state(ax, state, title):
    """Draw a Tower of Hanoi configuration.

    state : tuple length N, state[i] = peg (0/1/2) for disk i+1 (1 = smallest).
    """
    top_y = BASE_Y + PEG_H
    title_y = top_y + 0.04
    ax.set_xlim(0, 1)
    ax.set_ylim(BASE_Y - 0.07, title_y + 0.01)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title placed just above the peg tops in data coordinates
    ax.text(0.5, title_y, title, ha="center", va="bottom", fontsize=12)

    # Base (zorder=1: behind disks)
    ax.plot([0.03, 0.97], [BASE_Y, BASE_Y], color="black", lw=2,
            solid_capstyle="round", zorder=1)

    # Build stacks: iterate largest disk first so they sit at the bottom
    peg_stacks = {0: [], 1: [], 2: []}
    for disk_idx in range(len(state) - 1, -1, -1):   # 2, 1, 0  (large → small)
        peg_stacks[state[disk_idx]].append(disk_idx)

    # Draw disks (zorder=2: above base, below pegs)
    for peg, disk_list in peg_stacks.items():
        px = PEG_X[peg]
        for layer, disk_idx in enumerate(disk_list):
            w = DISK_W[disk_idx]
            rect = patches.FancyBboxPatch(
                (px - w / 2, BASE_Y + layer * DISK_H),
                w,
                DISK_H * DISK_FILL,
                boxstyle="round,pad=0.005",
                facecolor=DISK_COLORS[disk_idx],
                edgecolor="white",
                linewidth=1.2,
                zorder=2,
            )
            ax.add_patch(rect)

    # Pegs and peg labels (zorder=3: on top, shows through disk centres)
    for i, px in enumerate(PEG_X):
        ax.plot([px, px], [BASE_Y, top_y], color="black", lw=2,
                solid_capstyle="round", zorder=3)
        ax.text(px, BASE_Y - 0.05, str(i), ha="center", va="top", fontsize=11)


# --- Build figure ---
os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)

fig = plt.figure(figsize=(6.0, 1.6))
gs = fig.add_gridspec(1, 3, width_ratios=[5, 1, 5], wspace=0.0)

ax_start = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1])
ax_goal = fig.add_subplot(gs[2])

# Start: all disks on peg 0  →  state = (0, 0, 0)
draw_state(ax_start, (0, 0, 0), "Start")

# Arrow
ax_mid.axis("off")
ax_mid.text(0.5, 0.52, r"$\longrightarrow$", ha="center", va="center",
            fontsize=26, transform=ax_mid.transAxes)

# Goal: all disks on peg 2  →  state = (2, 2, 2)
draw_state(ax_goal, (2, 2, 2), "Goal")

out_path = os.path.join(os.path.dirname(__file__), "hanoi_schematic.pdf")
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
plt.close()
print(f"Saved {out_path}")
