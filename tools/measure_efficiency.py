#!/usr/bin/env python3
"""Measure wall-clock and peak memory per adaptation iteration (Table
efficiency). Example (A100 40GB, paper setup):

    python tools/measure_efficiency.py --config configs/tcrt/office_home.yaml \
        --iterations 20 --device cuda

The trainer is warmed up on random features so the numbers reflect the
adaptation-stage cost (certificate, anchors, graph) on real geometry.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from main import load_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from src.clip_backend import build_clip_backend
    from src.trainer import FeatureSampler, TCRTTrainer

    torch.manual_seed(0)
    backend = build_clip_backend(cfg, device)
    trainer = TCRTTrainer(cfg, backend, device)
    K = cfg["num_classes"]
    d = backend.d
    z_s = torch.randn(4096, d, device=device)
    y_s = torch.randint(0, K, (4096,), device=device)
    z_t = torch.randn(4096, d, device=device)
    # Fit the pieces the adaptation stage needs.
    trainer.energy_bins.fit(torch.cat([z_s[:1000].norm(dim=1), z_t[:1000].norm(dim=1)]).cpu())
    trainer.anchors.initialize_from_source(torch.randn(512, 16, device=device), y_s[:512])
    ss = FeatureSampler(z_s, y_s, cfg["batch_size"], args.iterations, 0)
    st = FeatureSampler(z_t, None, cfg["batch_size"], args.iterations, 0)
    # Warm up the backend.
    T = trainer._text_prototypes([f"class_{i}" for i in range(K)])
    for _ in range(3):
        z_b, y_b = next(ss)
        z_tb, _ = next(st)
        fs = trainer._project(z_b, T, y_b, "s")
        fs["y"] = y_b
        ft = trainer._project(z_tb, T, None, "t")
        loss, _, _ = trainer._adaptation_loss(fs, ft, T, 0.6, 0)
        loss.backward()
        trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iterations):
        z_b, y_b = next(ss)
        z_tb, _ = next(st)
        fs = trainer._project(z_b, T, y_b, "s")
        fs["y"] = y_b
        ft = trainer._project(z_tb, T, None, "t")
        loss, _, _ = trainer._adaptation_loss(fs, ft, T, 0.6, 0)
        loss.backward()
        trainer.optimizer.step()
    dt = (time.time() - t0) / args.iterations
    print(f"per-iteration: {dt * 1000:.1f} ms")
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"trainable parameters: {sum(p.numel() for p in trainer.opt_params) / 1e6:.4f} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
