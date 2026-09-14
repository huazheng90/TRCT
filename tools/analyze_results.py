#!/usr/bin/env python3
"""Aggregate results.json files into the paper's summary tables.

Usage:
    python tools/analyze_results.py --outputs outputs/office31 outputs/office_home \
        outputs/visda outputs/domainnet

Prints a Markdown table in the style of Table controlled_results and the
paired 95% intervals (Appendix paired_uncertainty) for each benchmark whose
config provides reference scores.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np


def load_results(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", nargs="+", required=True)
    args = p.parse_args()

    print("| Benchmark | Mean | Std | Per-seed scores |")
    print("|---|---|---|---|")
    for out in args.outputs:
        r = load_results(os.path.join(out, "results.json"))
        per_seed = ", ".join(f"{v:.2f}" for v in r["per_seed_benchmark_scores"])
        print(f"| {r['benchmark']} | {r['mean']:.2f} | {r['std']:.2f} | {per_seed} |")
        if "paired_interval" in r:
            pi = r["paired_interval"]
            print(f"  - paired 95% interval vs {r.get('reference', 'ref')}: "
                  f"gap {pi['gap']:.2f}  [{pi['ci_lo']:.2f}, {pi['ci_hi']:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
