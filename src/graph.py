"""Certificate-driven bipartite graph spectral regularization.

Implements Eq. anchor_affinity / active_anchors / bipartite_laplacian /
spectral_tail. The normalized graph is never explicitly materialized: every
graph multiplication costs O(|I_t| M) via the sparse bipartite structure
(``bar W = D_x^{-1/2} Q D_a^{-1} Q^T D_x^{-1/2}``). The spectral filter is
``g(lambda) = lambda^p`` with ``p in {1, 2}`` evaluated by one or two repeated
graph multiplications, without eigendecomposition.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.model import AnchorBank


def build_certified_graph(
    h_t: torch.Tensor,        # (n, r) semantic coordinates of certified target samples
    y_t: torch.Tensor,        # (n,) current target pseudo-labels
    anchors: AnchorBank,
    tau_g: float,
    eps_deg: float = 1e-6,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor]]:
    """Return (Q_active, inv_sqrt_dx, inv_da, active_anchor_ids).

    ``active_anchor_ids`` maps each active column to its global anchor index
    (needed by the diagnostics to recover anchor classes). Zero-degree anchors
    are removed before normalization; samples keep positive row degree by
    construction. Returns (None, ...) when the certified set is empty.
    """
    n = h_t.shape[0]
    if n == 0:
        return None, None, None, None
    Q, anchor_ids = anchors.affinity(h_t, y_t, tau_g)   # (n, M_c) per-class softmax
    M = anchors.K * anchors.M_c
    Q_full = torch.zeros(n, M, device=h_t.device)
    Q_full.scatter_(1, anchor_ids, Q)
    col_sum = Q_full.sum(dim=0)                         # (M,)
    active = col_sum > 0
    Q_active = Q_full[:, active]                        # (n, M')
    if Q_active.shape[1] == 0:
        return None, None, None, None
    d_x = Q_active.sum(dim=1).clamp(min=eps_deg)
    d_a = Q_active.sum(dim=0).clamp(min=eps_deg)
    inv_sqrt_dx = d_x.pow(-0.5)
    inv_da = d_a.pow(-1.0)
    active_ids = torch.arange(M, device=h_t.device)[active]
    return Q_active, inv_sqrt_dx, inv_da, active_ids


def spectral_tail_loss(
    P: torch.Tensor,          # (n, K) adaptive posteriors of certified samples
    Q: torch.Tensor,          # (n, M') active affinity
    inv_sqrt_dx: torch.Tensor,
    inv_da: torch.Tensor,
    p: int = 1,
) -> torch.Tensor:
    """tr(P^T g(L_c) P) / n with g(lambda) = lambda^p, p in {1, 2}."""
    n = P.shape[0]
    if n == 0:
        return torch.zeros((), device=P.device)
    Y = inv_sqrt_dx.unsqueeze(1) * P                    # (n, K) = D_x^-1/2 P
    WY = inv_sqrt_dx.unsqueeze(1) * (Q @ (inv_da.unsqueeze(1) * (Q.t() @ Y)))
    LY = Y - WY                                         # L_c P
    if p == 2:
        WLY = inv_sqrt_dx.unsqueeze(1) * (Q @ (inv_da.unsqueeze(1) * (Q.t() @ LY)))
        LY = LY - WLY                                   # L_c^2 P
    return (P * LY).sum() / n


def graph_purity_and_smoothing(
    h_t: torch.Tensor,        # (n, r) semantic coordinates of the selected nodes
    y_t: torch.Tensor,        # (n,) pseudo-labels used to build the graph
    labels_t: torch.Tensor,   # (n,) withheld target labels (post-hoc only)
    anchors: AnchorBank,
    tau_g: float,
    eps_deg: float = 1e-6,
) -> Tuple[float, float, float]:
    """Post-hoc graph quality: weighted purity, effective degree, smoothed acc.

    Only used by the diagnostics (never during training). ``labels_t`` must be
    withheld and are opened only post hoc.
    """
    n = h_t.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    Q, inv_sqrt_dx, inv_da, active_ids = build_certified_graph(h_t, y_t, anchors, tau_g, eps_deg)
    if Q is None:
        return 0.0, 0.0, 0.0
    anchor_classes = active_ids // anchors.M_c           # (M',)
    total_w = float(Q.sum().item())
    same = 0.0
    for m in range(Q.shape[1]):
        c = int(anchor_classes[m])
        same += float(Q[labels_t == c, m].sum().item())
    purity = same / max(total_w, 1e-12)
    eff_deg = float(Q.sum(dim=1).mean().item())
    smooth = torch.zeros(n, anchors.K, device=h_t.device)
    for m in range(Q.shape[1]):
        c = int(anchor_classes[m])
        if c >= 0:
            smooth[:, c] += Q[:, m]
    smoothed_pred = smooth.argmax(dim=1)
    acc = float((smoothed_pred == labels_t).float().mean().item())
    return purity, eff_deg, acc
