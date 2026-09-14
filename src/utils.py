"""Shared utilities: seeding, logging, metrics, and paired statistics.

Everything here is deterministic and side-effect free except for logging.
Seed management follows the paper's five matched seeds: benchmark score for a
seed is the mean over directed transfers, and comparisons use the five
unrounded seedwise paired differences.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger("tcrt")


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism knobs; left off by default because they slow training.
    # torch.backends.cudnn.deterministic = True


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Number of tensors updated by the optimizer (excludes frozen backbone)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def mean_class_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                        num_classes: int) -> float:
    """Per-class recall averaged over classes (VisDA-2017 mAcc)."""
    preds = logits.argmax(dim=1)
    per_class = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).float().mean()
    return float(per_class.mean().item())


def paired_interval(tcrt_scores: Sequence[float],
                    other_scores: Sequence[float]) -> Dict[str, float]:
    """Two-sided 95% Student-t interval (df=4) for the seedwise mean gap.

    ``tcrt_scores`` and ``other_scores`` are the five unrounded benchmark
    scores (mean over transfers per seed). Returns the displayed mean gap and
    the interval endpoints in percentage points.
    """
    assert len(tcrt_scores) == len(other_scores) == 5, "five matched seeds required"
    diffs = np.asarray(tcrt_scores, dtype=np.float64) - np.asarray(other_scores, dtype=np.float64)
    mean_d = float(diffs.mean())
    if len(diffs) < 2:
        return {"gap": mean_d, "ci_lo": mean_d, "ci_hi": mean_d}
    sd = float(diffs.std(ddof=1))
    t = 2.7764451051977987  # t_{0.975, 4}
    half = t * sd / math.sqrt(len(diffs))
    return {"gap": mean_d, "ci_lo": mean_d - half, "ci_hi": mean_d + half}


def point_biserial(scores: np.ndarray, labels: np.ndarray) -> float:
    """Point-biserial correlation between a score and a binary label."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n1 = float(labels.sum())
    n0 = float(len(labels) - n1)
    if n0 == 0 or n1 == 0:
        return 0.0
    mu1 = scores[labels == 1].mean()
    mu0 = scores[labels == 0].mean()
    sd = scores.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float((mu1 - mu0) / sd * math.sqrt(n1 * n0 / (len(labels) ** 2)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg = np.zeros(len(np.unique(scores)), dtype=np.float64)
    for i, c in enumerate(counts):
        avg[i] = ranks[inverse == i].mean()
    rank_avg = avg[inverse]
    u = rank_avg[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auroc_fast(scores: np.ndarray, labels: np.ndarray) -> float:
    """Faster AUROC via rank-based Mann-Whitney (no tie averaging loop)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def aurc(scores: np.ndarray, labels: np.ndarray, coverage_grid: Optional[np.ndarray] = None) -> float:
    """Area under the risk-coverage curve; lower is better."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_err = 1.0 - labels[order]
    n = len(sorted_err)
    if coverage_grid is None:
        coverage_grid = np.linspace(0.05, 1.0, 20)
    risks = []
    for cov in coverage_grid:
        k = max(1, int(round(cov * n)))
        risks.append(float(sorted_err[:k].mean()))
    risks = np.asarray(risks)
    return float(np.trapezoid(risks, coverage_grid) / (coverage_grid[-1] - coverage_grid[0]))


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transfer_name(src: str, tgt: str) -> str:
    return f"{src}->{tgt}"
