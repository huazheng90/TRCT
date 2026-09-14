"""CPU smoke test for the full TCRT pipeline on a mock CLIP backend.

Run:  python tests/test_smoke.py

This validates the data flow, all loss terms, the alternating solver, the
certificate, the graph, checkpoint save/load, evaluation, and a diagnostics
entry point — without downloading CLIP weights or any dataset images.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed  # noqa: E402


def make_cfg(tmp: str, num_classes: int = 10) -> dict:
    return {
        "backend": "mock",
        "feat_dim": 64,
        "num_classes": num_classes,
        "seed": 0,
        "seeds": [0, 1],
        "n_ctx": 4,
        "text_logit_scale": 20.0,
        "semantic_rank": 16,
        "rank_eps": 1e-3,
        "residual_rank": 6,
        "anchors_per_class": 2,
        "rho": 0.1,
        "eta": 0.1,
        "kappa_cert": 0.0,
        "zeta": 0.1,
        "kappa_donor": 0.05,
        "kappa_r": 2.0,
        "j_min": 1,
        "queue_capacity": 512,
        "energy_bins": 4,
        "pi_min": 0.4,
        "pi_max": 0.8,
        "lambda_cf": 1.0,
        "anchor_momentum": 0.99,
        "tau_g": 5.0,
        "spectral_p": 1,
        "eps_deg": 1e-6,
        "lr": 3e-3,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "batch_size": 8,
        "iterations_per_epoch": 2,
        "warmup_epochs": 1,
        "adapt_epochs": 1,
        "lambda_ent": 1.0,
        "lambda_fac": 1.0,
        "lambda_priv": 1.0,
        "lambda_leak": 0.1,
        "lambda_cert": 1.0,
        "lambda_tail": 1.0,
        "lambda_sup": 1.0,
        "use_disentangle": True,
        "use_counterfactual": True,
        "alpha": 1.0, "beta": 1e-3, "xi": 1e-3, "gamma": 1e-2,
        "j_alt": 3, "lr_block": 0.1,
        "output_dir": os.path.join(tmp, "out"),
        "checkpoint": os.path.join(tmp, "out", "model.pt"),
        "log_every": 1,
        "device": "cpu",
        "transfers": [["src", "tgt"]],
    }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="tcrt_smoke_")
    set_seed(0)
    device = torch.device("cpu")
    cfg = make_cfg(tmp)
    K = cfg["num_classes"]

    from src.clip_backend import build_clip_backend
    from src.trainer import TCRTTrainer

    backend = build_clip_backend(cfg, device)
    class_names = [f"class_{i}" for i in range(K)]

    # Synthetic features: source separable, target shifted.
    g = torch.Generator().manual_seed(7)
    z_s = torch.randn(200, cfg["feat_dim"], generator=g)
    y_s = torch.randint(0, K, (200,), generator=g)
    z_s = torch.nn.functional.normalize(z_s, dim=-1)
    z_t = torch.randn(150, cfg["feat_dim"], generator=g) * 1.5
    z_t = torch.nn.functional.normalize(z_t, dim=-1)
    y_t = torch.randint(0, K, (150,), generator=g)

    trainer = TCRTTrainer(cfg, backend, device)
    trainer.fit(class_names, z_s, y_s, z_t, save_path=cfg["checkpoint"])
    assert not torch.isnan(trainer.optimizer.param_groups[0]["params"][0].grad).any(), "NaN gradient"

    # Save / load round-trip.
    trainer.save(cfg["checkpoint"])
    trainer2 = TCRTTrainer(cfg, backend, device)
    trainer2.load(cfg["checkpoint"], load_optimizer=False)
    T = backend.text_prototypes(class_names)
    acc = trainer2.evaluate(class_names, T, z_t, y_t)
    print(f"[smoke] evaluation acc={acc:.4f}")
    assert 0.0 <= acc <= 100.0

    # Diagnostics smoke: ranking + coverage risk on the tiny mock data.
    from src import diagnostics as D
    import data as data_mod
    # Place a checkpoint at the layout diagnostics expect.
    ckpt_dir = os.path.join(tmp, "src->tgt", "seed0")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save(os.path.join(ckpt_dir, "model.pt"))
    cfg["seeds"] = [0]
    cfg["output_dir"] = tmp
    # Build a minimal fake data dict so diagnostics can run without images.
    class _DM:
        def __init__(self, cfg=None, backend=None, device=None):
            pass

        def prepare(self, *a, **k):
            return {"z_s": z_s, "y_s": y_s, "z_t": z_t, "y_t": y_t,
                    "z_t_test": None, "y_t_test": None, "class_names": class_names}
    data_mod.DataModule = _DM
    D.build_clip_backend = lambda c, d: backend
    res = D.ranking_diagnostic(cfg, device)
    print("[smoke] ranking diagnostic keys:", list(res.keys()))
    res2 = D.coverage_risk_diagnostic(cfg, device)
    print("[smoke] coverage-risk curves:", {k: len(v) for k, v in res2.items()})
    res3 = D.additivity_diagnostic(cfg, device)
    print("[smoke] additivity:", res3)
    res4 = D.property_checks(cfg, device, n_pairs=50)
    print("[smoke] property checks:", res4)
    res5 = D.graph_quality_diagnostic(cfg, device)
    print("[smoke] graph quality keys:", list(res5.keys()))

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
