#!/usr/bin/env python3
"""Reconstruct LLM result files from a cancelled job's log output.

When a SLURM job is killed before save_results() completes, the per-episode
data is still visible in the .err log.  This script parses those lines and
writes the same .pt and .json files that llm_hanoi_eval.py would have produced.

Usage:
    python llm_eval/recover_from_log.py \\
        --log    logs/llm_qwen25_7b_s0_zero_shot_9161383.err \\
        --timestamp 1748875208 \\
        --difficulty ES \\
        --model_label qwen25_7b \\
        --prompting zero_shot
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--log",         required=True, help="Path to the .err log file")
parser.add_argument("--timestamp",   required=True)
parser.add_argument("--difficulty",  required=True, choices=["ES", "MS", "LS", "RandState"])
parser.add_argument("--model_label", required=True)
parser.add_argument("--prompting",   required=True)
args = parser.parse_args()

# --------------------------------------------------------------------------
# Parse log lines like:
#   ep   1/50  error= 193  solved=False  illegal_rate=0.27  parse_fail=0
# --------------------------------------------------------------------------
pattern = re.compile(
    r"ep\s+(\d+)/\d+\s+"
    r"error=\s*(-?\d+)\s+"
    r"solved=(\w+)\s+"
    r"illegal_rate=([\d.]+)\s+"
    r"parse_fail=(\d+)"
)

episodes = []
with open(args.log) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            episodes.append({
                "ep":           int(m.group(1)),
                "error":        int(m.group(2)),
                "solved":       m.group(3).lower() == "true",
                "illegal_rate": float(m.group(4)),
                "parse_failures": int(m.group(5)),
            })

if not episodes:
    print("ERROR: no episode lines found in log. Check the path / format.")
    sys.exit(1)

print(f"Recovered {len(episodes)} episodes from log.")

# --------------------------------------------------------------------------
# Aggregate (same logic as _aggregate() in llm_hanoi_eval.py)
# --------------------------------------------------------------------------
errors        = np.array([e["error"]        for e in episodes], dtype=float)
solve_rates   = np.array([e["solved"]       for e in episodes], dtype=float)
illegal_rates = np.array([e["illegal_rate"] for e in episodes], dtype=float)

n = len(episodes)
se = lambda x: float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

agg = {
    "mean_error":        float(errors.mean()),
    "std_error":         float(errors.std(ddof=1)) if n > 1 else 0.0,
    "se_error":          se(errors),
    "solve_rate":        float(solve_rates.mean()),
    "se_solve_rate":     se(solve_rates),
    "mean_illegal_rate": float(illegal_rates.mean()),
    "se_illegal_rate":   se(illegal_rates),
    "n_episodes":        n,
    "recovered_from_log": args.log,
    "episodes":          episodes,
}

print(f"  mean_error={agg['mean_error']:.2f}  se={agg['se_error']:.2f}"
      f"  solve_rate={agg['solve_rate']:.2%}"
      f"  illegal_rate={agg['mean_illegal_rate']:.2%}")

# --------------------------------------------------------------------------
# Save (same format as save_results() in llm_hanoi_eval.py)
# --------------------------------------------------------------------------
save_dir  = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp, args.difficulty)
os.makedirs(save_dir, exist_ok=True)
file_stem = f"LLM_{args.model_label}_{args.prompting}"

mean_e = agg["mean_error"]
se_e   = agg["se_error"]

acc_tensor = torch.tensor([[0.0, mean_e]])
torch.save(acc_tensor, os.path.join(save_dir, file_stem + "_actingAccuracy.pt"))

err_tensor = torch.tensor([[0.0, mean_e, se_e]])
torch.save(err_tensor, os.path.join(save_dir, file_stem + "_actingAccuracy_error.pt"))

json_path = os.path.join(save_dir, file_stem + "_results.json")
with open(json_path, "w") as f:
    json.dump(agg, f, indent=2)

print(f"Saved to: {save_dir}/")
print(f"  {file_stem}_actingAccuracy.pt")
print(f"  {file_stem}_actingAccuracy_error.pt")
print(f"  {file_stem}_results.json")
