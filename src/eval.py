"""Evaluation driver: controlled transductive / inductive benchmarks.

Runs all directed transfers of a benchmark with the five matched seeds and
produces:
- per-transfer, per-seed results (archived as JSON),
- the benchmark score A_s = mean over transfers per seed,
- benchmark mean +/- sample std, and paired 95% intervals vs a reference
  method when provided.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.trainer import TCRTTrainer
from src.utils import LOGGER, paired_interval, set_seed, transfer_name


def run_transfer(cfg: dict, backend, data: dict, device: torch.device,
                 eval_mode: str = "accuracy") -> Tuple[float, Dict[str, float]]:
    """Train TCRT on one transfer; return (score, per-seed acc) for one seed."""
    set_seed(cfg["seed"])
    trainer = TCRTTrainer(cfg, backend, device)
    history = trainer.fit(
        class_names=data["class_names"],
        z_s=data["z_s"], y_s=data["y_s"],
        z_t=data["z_t"],
        z_t_test=data.get("z_t_test"), y_t_test=data.get("y_t_test"),
        eval_every=cfg.get("eval_every", 0),
        log_every=cfg.get("log_every", 100),
        save_path=cfg.get("checkpoint"),
    )
    T = trainer._text_prototypes(data["class_names"])
    if data.get("z_t_test") is not None:
        acc = trainer.evaluate(data["class_names"], T, data["z_t_test"], data["y_t_test"])
    else:
        acc = trainer.evaluate(data["class_names"], T, data["z_t"], data["y_t"])
    if eval_mode == "macc":
        # mean class accuracy on the full target set (VisDA).
        acc = _macc(trainer, data["class_names"], T, data["z_t"], data["y_t"])
    return acc, {"acc": acc}


def _macc(trainer, class_names, T, z_t, y_t) -> float:
    device = z_t.device
    V = trainer.factorization.V()
    C = T @ V
    preds = []
    for i in range(0, z_t.shape[0], 256):
        z = z_t[i: i + 256].to(device)
        h = z @ V
        p = torch.softmax(trainer.text_logit_scale * h @ C.t(), dim=-1)
        preds.append(p.argmax(dim=1).cpu())
    preds = torch.cat(preds)
    labels = y_t.cpu()
    K = trainer.K
    per_class = []
    for c in range(K):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append(float((preds[mask] == c).float().mean().item()))
    return float(np.mean(per_class))


def run_benchmark(cfg: dict, device: torch.device) -> Dict:
    """Run all transfers x five seeds; archive results."""
    from data import DataModule
    from src.clip_backend import build_clip_backend

    transfers = cfg["transfers"]
    seeds = cfg["seeds"] if isinstance(cfg["seeds"], (list, tuple)) else [cfg["seeds"]]
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)

    per_transfer = {}
    bench_scores: Dict[str, List[float]] = {}
    for src, tgt in transfers:
        tname = transfer_name(src, tgt)
        per_transfer[tname] = {}
        for seed in seeds:
            cfg["seed"] = seed
            data = dm.prepare(src, tgt, inductive=cfg.get("inductive", False))
            cfg["checkpoint"] = os.path.join(
                cfg["output_dir"], tname, f"seed{seed}", "model.pt")
            t0 = time.time()
            acc, _ = run_transfer(cfg, backend, data, device,
                                  eval_mode=cfg.get("eval_mode", "accuracy"))
            per_transfer[tname][str(seed)] = {"acc": round(acc, 4),
                                              "secs": round(time.time() - t0, 1)}
            LOGGER.info("[%s] seed %d acc=%.4f", tname, seed, acc)
        bench_scores[tname] = [per_transfer[tname][str(s)]["acc"] for s in seeds]

    # Benchmark score per seed = mean over transfers.
    per_seed = []
    for i, seed in enumerate(seeds):
        vals = [per_transfer[tname][str(seed)]["acc"] for tname in per_transfer]
        per_seed.append(float(np.mean(vals)))

    summary = {
        "benchmark": cfg["benchmark"],
        "seeds": seeds,
        "per_transfer": per_transfer,
        "per_seed_benchmark_scores": per_seed,
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "iterations_per_epoch": cfg.get("iterations_per_epoch", 1000),
    }
    if cfg.get("reference_scores") is not None:
        ref = cfg["reference_scores"]
        pi = paired_interval(per_seed, ref)
        summary["reference"] = cfg.get("reference_name", "reference")
        summary["paired_interval"] = pi
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out = os.path.join(cfg["output_dir"], "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("benchmark %s mean=%.4f std=%.4f -> %s",
                cfg["benchmark"], summary["mean"], summary["std"], out)
    return summary
