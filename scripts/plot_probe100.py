#!/usr/bin/env python3
"""
Plot average pass_rate vs g_level for probe100 results.
Produces two subplots: one for 'prefix' mode, one for 'hint' mode.

Usage:
    python scripts/plot_probe100.py [path_to_pass_rates.jsonl]

If no path given, defaults to the latest probe100 output.
"""
import json
import sys
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pass_rates(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "./outputs/probe100/Qwen3-1.7B_PASS32/per_question_pass_PASS32.jsonl"

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    items = load_pass_rates(path)
    print(f"Loaded {len(items)} entries from {path}")

    modes = sorted(set(item["guidance_mode"] for item in items))

    # group: mode -> g_level -> [pass_rate, ...]
    grouped = {mode: defaultdict(list) for mode in modes}
    for item in items:
        grouped[item["guidance_mode"]][item["g_level"]].append(item["pass_rate"])

    n_questions = len(set(item["index"] for item in items))
    pass_n = 0
    for item in items:
        if item["pass_rate"] > 0:
            pass_n = int(round(item["pass_count"] / item["pass_rate"]))
            break

    colors = {"prefix": "#2563EB", "hint": "#E11D48"}
    markers = {"prefix": "s", "hint": "o"}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode in modes:
        g_levels = sorted(grouped[mode].keys())
        avg_rates = [sum(grouped[mode][g]) / len(grouped[mode][g]) for g in g_levels]

        g_levels = [0.0] + g_levels
        avg_rates = [0.0] + avg_rates

        ax.plot(g_levels, avg_rates,
                marker=markers.get(mode, "o"), linewidth=2.2, markersize=7,
                color=colors.get(mode, None), label=mode)

        for g, r in zip(g_levels, avg_rates):
            offset_y = 12 if mode == "hint" else -18
            ax.annotate(f"{r:.3f}", (g, r), textcoords="offset points",
                        xytext=(0, offset_y), ha="center", fontsize=9,
                        color=colors.get(mode, "black"))

    ax.set_xlabel("g_level (guidance fraction)", fontsize=12)
    ax.set_ylabel("Average Pass Rate", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(title="Guidance Mode", fontsize=11, title_fontsize=11)

    fig.suptitle(f"Probe100: Average Pass Rate vs Guidance Level\n"
                 f"({n_questions} questions, Pass@{pass_n})",
                 fontsize=14)
    fig.tight_layout()

    out_dir = os.path.dirname(path)
    out_path = os.path.join(out_dir, "probe100_pass_rate_vs_g.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
