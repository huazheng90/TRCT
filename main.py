"""TCRT entry point.

Commands
--------
python main.py benchmark  --config configs/tcrt/office_home.yaml
    Run every directed transfer x every seed of the benchmark and archive
    per-transfer / per-seed results plus the benchmark summary.

python main.py run        --config configs/tcrt/office_home.yaml --src Ar --tgt Cl
    Run a single transfer over the configured seeds.

python main.py diagnose   --config ... --kind ranking|coverage_risk|stage|graph|probe|additivity|transport|properties
    Post-hoc mechanism analysis on saved checkpoints (no retraining).

python main.py count_params --config ...
    Report trainable parameter count (the Table efficiency row for TCRT).

python main.py extract    --config ...
    Pre-extract frozen CLIP image features to the cache directory.

Any config field can be overridden with ``--set key=value``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import LOGGER, count_trainable_parameters, setup_logging  # noqa: E402


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    if overrides:
        for kv in overrides:
            key, _, val = kv.partition("=")
            # try typed parsing
            try:
                cfg[key] = yaml.safe_load(val)
            except Exception:
                cfg[key] = val
    # Seed list normalization.
    if isinstance(cfg.get("seeds"), int):
        cfg["seeds"] = [cfg["seeds"]]
    if isinstance(cfg.get("transfers"), (list, tuple)) and \
            isinstance(cfg["transfers"][0], str):
        cfg["transfers"] = [cfg["transfers"]]
    cfg.setdefault("output_dir", os.path.join("outputs", cfg.get("experiment_name", "tcrt")))
    return cfg


def _device(cfg: Dict) -> torch.device:
    if cfg.get("device") in ("cpu", "gpu"):
        return torch.device("cpu") if cfg["device"] == "cpu" else torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_benchmark(cfg: Dict, args) -> int:
    from src.eval import run_benchmark
    summary = run_benchmark(cfg, _device(cfg))
    print("\n=== benchmark summary ===")
    print(yaml.safe_dump({"benchmark": summary["benchmark"],
                          "mean": summary["mean"], "std": summary["std"],
                          "per_seed": summary["per_seed_benchmark_scores"]},
                         sort_keys=False))
    return 0


def cmd_run(cfg: Dict, args) -> int:
    from data import DataModule
    from src.clip_backend import build_clip_backend
    from src.eval import run_transfer

    device = _device(cfg)
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    cfg["checkpoint"] = os.path.join(cfg["output_dir"], f"{args.src}->{args.tgt}",
                                     f"seed{cfg['seeds'][0]}", "model.pt")
    data = dm.prepare(args.src, args.tgt, inductive=cfg.get("inductive", False))
    acc, _ = run_transfer(cfg, backend, data, device, eval_mode=cfg.get("eval_mode", "accuracy"))
    print(f"acc={acc:.4f} (seed {cfg['seeds'][0]}, {args.src}->{args.tgt})")
    return 0


def cmd_diagnose(cfg: Dict, args) -> int:
    from src import diagnostics as D
    device = _device(cfg)
    kinds = args.kind.split(",")
    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    for kind in kinds:
        t0 = time.time()
        if kind == "ranking":
            res = D.ranking_diagnostic(cfg, device)
        elif kind == "coverage_risk":
            res = D.coverage_risk_diagnostic(cfg, device)
        elif kind == "stage":
            res = D.stage_diagnostic(cfg, device)
        elif kind == "graph":
            res = D.graph_quality_diagnostic(cfg, device)
        elif kind == "probe":
            res = D.factorization_probe_diagnostic(cfg, device)
        elif kind == "additivity":
            res = D.additivity_diagnostic(cfg, device)
        elif kind == "transport":
            res = D.transport_plausibility_diagnostic(cfg, device)
        elif kind == "properties":
            res = D.property_checks(cfg, device)
        else:
            LOGGER.error("unknown diagnostic kind: %s", kind)
            return 1
        path = os.path.join(out_dir, f"{kind}.json")
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        LOGGER.info("%s diagnostic -> %s (%.1fs)", kind, path, time.time() - t0)
        print(yaml.safe_dump(res, sort_keys=False))
    return 0


def cmd_count_params(cfg: Dict, args) -> int:
    from src.clip_backend import build_clip_backend
    from src.trainer import TCRTTrainer
    trainer = TCRTTrainer(cfg, build_clip_backend(cfg, _device(cfg)), _device(cfg))
    n = sum(p.numel() for p in trainer.opt_params)
    print(f"trainable parameters: {n / 1e6:.4f} M")
    return 0


def cmd_extract(cfg: Dict, args) -> int:
    from data import DataModule
    from src.clip_backend import build_clip_backend
    device = _device(cfg)
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    for src, tgt in cfg["transfers"]:
        dm.prepare(src, tgt, inductive=cfg.get("inductive", False))
    print("feature cache ready")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="TCRT: Text-Certified Residual Transport")
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn in [("benchmark", cmd_benchmark), ("run", cmd_run),
                     ("diagnose", cmd_diagnose), ("count_params", cmd_count_params),
                     ("extract", cmd_extract)]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--set", action="append", default=None,
                        help="override config key=value")
        if name == "run":
            sp.add_argument("--src", required=True)
            sp.add_argument("--tgt", required=True)
        if name == "diagnose":
            sp.add_argument("--kind", required=True,
                            help="ranking,coverage_risk,stage,graph,probe,additivity,transport,properties")
        sp.set_defaults(func=fn)

    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    setup_logging(cfg.get("log_level", "INFO"),
                  os.path.join(cfg["output_dir"], "train.log"))
    LOGGER.info("config: %s", args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
