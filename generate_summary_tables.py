#!/usr/bin/env python3
"""Generate CSV summary tables from MuZero and LLM experimental data.

Usage:
    python generate_summary_tables.py --timestamp 1748875208
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np
import torch


# ── Constants ─────────────────────────────────────────────────────────────────

MUZERO_CONDITIONS = [
    ("Muzero", "MuZero"),
    ("ResetLatentPol", "Policy Ablated"),
    ("ResetLatentVal", "Value Ablated"),
    ("ResetLatentRwd", "Reward Ablated"),
    ("ResetLatentVal_ResetLatentRwd", "Value + Reward Ablated"),
]

DIFFICULTIES = [
    ("ES", "Far"),
    ("MS", "Moderate"),
    ("LS", "Close"),
]


# ── Data loading (reused from existing plotting code) ─────────────────────────

def load_accuracy(directory: str, label: str) -> np.ndarray | None:
    """Load *_actingAccuracy.pt with optional variance column."""
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


def load_llm_json(path: str) -> dict | None:
    """Load a single LLM results JSON file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_muzero_csv(root_dir: str, output_path: str, muzero_runs: int):
    """Write muzero_summary.csv with all conditions × difficulties × sim counts."""
    se_factor = 1.0 / math.sqrt(muzero_runs)
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
                    "condition": label,
                    "display_name": display_name,
                    "difficulty": diff_name,
                    "difficulty_code": diff_dir,
                    "n_sims": n_sims,
                    "mean_error": f"{mean_error:.4f}",
                    "std_error": f"{std_error:.4f}" if has_std else "",
                    "se_error": f"{se_error:.4f}" if has_std else "",
                })

    fieldnames = ["condition", "display_name", "difficulty", "difficulty_code",
                  "n_sims", "mean_error", "std_error", "se_error"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {output_path}")


def write_llm_csv(root_dir: str, output_path: str):
    """Write llm_summary.csv by auto-discovering LLM_*_results.json files."""
    rows = []

    for diff_dir, diff_name in DIFFICULTIES:
        file_dir = os.path.join(root_dir, diff_dir)
        pattern = os.path.join(file_dir, "LLM_*_results.json")
        for json_path in sorted(glob.glob(pattern)):
            fname = os.path.basename(json_path)
            # Extract condition: "LLM_<condition>_results.json"
            condition = fname.replace("LLM_", "").replace("_results.json", "")

            data = load_llm_json(json_path)
            if data is None:
                continue

            rows.append({
                "condition": condition,
                "difficulty": diff_name,
                "difficulty_code": diff_dir,
                "mean_error": f"{data['mean_error']:.4f}",
                "se_error": f"{data['se_error']:.4f}",
                "solve_rate": f"{data['solve_rate']:.4f}",
                "se_solve_rate": f"{data.get('se_solve_rate', 0):.4f}",
                "illegal_rate": f"{data['mean_illegal_rate']:.4f}",
                "se_illegal_rate": f"{data['se_illegal_rate']:.4f}",
                "n_episodes": data.get("n_episodes", ""),
            })

    fieldnames = ["condition", "difficulty", "difficulty_code", "mean_error",
                  "se_error", "solve_rate", "se_solve_rate", "illegal_rate",
                  "se_illegal_rate", "n_episodes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {output_path}")


def write_combined_csv(root_dir: str, output_path: str, muzero_runs: int):
    """Write combined_summary.csv comparing MuZero (n_sims=150) with LLM."""
    se_factor = 1.0 / math.sqrt(muzero_runs)
    rows = []

    for diff_dir, diff_name in DIFFICULTIES:
        file_dir = os.path.join(root_dir, diff_dir)

        # MuZero at n_sims=150 (last row)
        for label, display_name in MUZERO_CONDITIONS:
            arr = load_accuracy(file_dir, label)
            if arr is None:
                continue
            has_std = arr.shape[1] > 2
            last = arr[-1]
            mean_error = last[1]
            std_e = last[2] if has_std else float("nan")
            se_e = std_e * se_factor if has_std else float("nan")
            rows.append({
                "agent_type": "MuZero",
                "condition": display_name,
                "difficulty": diff_name,
                "difficulty_code": diff_dir,
                "mean_error": f"{mean_error:.4f}",
                "se_error": f"{se_e:.4f}" if has_std else "",
                "solve_rate": "",
                "se_solve_rate": "",
                "illegal_rate": "",
                "se_illegal_rate": "",
            })

        # LLM conditions
        pattern = os.path.join(file_dir, "LLM_*_results.json")
        for json_path in sorted(glob.glob(pattern)):
            fname = os.path.basename(json_path)
            condition = fname.replace("LLM_", "").replace("_results.json", "")
            data = load_llm_json(json_path)
            if data is None:
                continue
            rows.append({
                "agent_type": "LLM",
                "condition": condition,
                "difficulty": diff_name,
                "difficulty_code": diff_dir,
                "mean_error": f"{data['mean_error']:.4f}",
                "se_error": f"{data['se_error']:.4f}",
                "solve_rate": f"{data['solve_rate']:.4f}",
                "se_solve_rate": f"{data.get('se_solve_rate', 0):.4f}",
                "illegal_rate": f"{data['mean_illegal_rate']:.4f}",
                "se_illegal_rate": f"{data['se_illegal_rate']:.4f}",
            })

    fieldnames = ["agent_type", "condition", "difficulty", "difficulty_code",
                  "mean_error", "se_error", "solve_rate", "se_solve_rate",
                  "illegal_rate", "se_illegal_rate"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV summary tables from experiment data"
    )
    parser.add_argument("--timestamp", required=True, help="Experiment timestamp directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: stats/Hanoi/<timestamp>/tables/)")
    parser.add_argument("--muzero_runs", type=int, default=5,
                        help="Number of MuZero runs for SD→SE conversion")
    args = parser.parse_args()

    root_dir = os.path.join("stats", "Hanoi", args.timestamp)
    if not os.path.isdir(root_dir):
        print(f"Error: directory not found: {root_dir}")
        return

    output_dir = args.output_dir or os.path.join(root_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)

    write_muzero_csv(root_dir, os.path.join(output_dir, "muzero_summary.csv"), args.muzero_runs)
    write_llm_csv(root_dir, os.path.join(output_dir, "llm_summary.csv"))
    write_combined_csv(root_dir, os.path.join(output_dir, "combined_summary.csv"), args.muzero_runs)

    print(f"\nAll tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
