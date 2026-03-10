#!/usr/bin/env python3
"""Scatter plot: mean parse failures per episode vs mean error, all LLM conditions.

This is a thin wrapper — the implementation lives in generate_all_figures.py.
"""

import argparse
import os
import sys

import matplotlib as mpl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from generate_all_figures import fig_parse_failures
from utils import set_plot_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default="1748875208")
    parser.add_argument("--no_latex", action="store_true")
    args = parser.parse_args()

    set_plot_style()
    if args.no_latex:
        mpl.rcParams["text.usetex"] = False

    root_dir = os.path.join(PROJECT_ROOT, "stats", "Hanoi", args.timestamp)
    fig_parse_failures(root_dir, args.timestamp)


if __name__ == "__main__":
    main()
