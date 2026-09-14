"""Residual donor queues, cross-domain transport, and the text certificate.

Implements the paper's certification pipeline:

- FIFO residual queues ``Q_d`` per domain (detached snapshots with class tag,
  energy bin, and text margin),
- the valid donor set ``J_i^in`` (same class, energy-matched, energy-ratio and,
  for target donors, text-margin constrained; Eq. donor_set),
- the tail transport discrepancy ``u_i = CVaR_rho KL(q_T(z_i) || q_T(z~_{i<-j}))``
  (Eq. transport_discrepancy),
- the certificate ``c_i = exp(-u_i/eta) * sigmoid((m_i - kappa)/zeta)``
  (Eq. certificate),
- class-wise top-``pi_e`` certified set ``I_t``.

Energy-bin boundaries are fixed from a warm-up reservoir so that queue entries
and current samples share the same map ``b(.)``.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ResidualQueue:
    """FIFO memory of detached residual snapshots for one domain."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._r: List[torch.Tensor] = []          # detached (d,)
        self._cls: List[int] = []
        self._bin: List[int] = []
        self._margin: List[float] = []

    def __len__(self) -> int:
        return len(self._r)

    def clear(self) -> None:
        self._r, self._cls, self._bin, self._margin = [], [], [], []

    def push(self, r: torch.Tensor, cls: torch.Tensor, bin_: torch.Tensor,
             margin: torch.Tensor) -> None:
        r = r.detach()
        margin = margin.detach()
        n = r.shape[0]
        self._r.extend([r[i] for i in range(n)])
        self._cls.extend([int(cls[i]) for i in range(n)])
        self._bin.extend([int(bin_[i]) for i in range(n)])
        self._margin.extend([float(margin[i]) for i in range(n)])
        if len(self._r) > self.capacity:
            self._r = self._r[-self.capacity:]
            self._cls = self._cls[-self.capacity:]
            self._bin = self._bin[-self.capacity:]
            self._margin = self._margin[-self.capacity:]

    def as_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor,
                                                        torch.Tensor, torch.Tensor]:
        if len(self._r) == 0:
            # Shape (0, 0): callers skip cat when a queue is empty.
            return (torch.zeros(0, 0, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, device=device))
        return (torch.stack(self._r).to(device),
                torch.tensor(self._cls, dtype=torch.long, device=device),
                torch.tensor(self._bin, dtype=torch.long, device=device),
                torch.tensor(self._margin, device=device))


class EnergyBins:
    """Fixed quantile bin boundaries for residual norms (from warm-up)."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.edges: Optional[np.ndarray] = None

    def fit(self, norms: torch.Tensor) -> None:
        norms = norms.detach().cpu().numpy()
        if norms.size == 0:
            self.edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            return
        qs = np.linspace(0.0, 100.0, self.n_bins + 1)
        self.edges = np.percentile(norms, qs)
        self.edges[0] = -np.inf
        self.edges[-1] = np.inf

    def assign(self, norms: torch.Tensor) -> torch.Tensor:
        norms = norms.detach().cpu().numpy()
        idx = np.digitize(norms, self.edges[1:-1], right=True)
        return torch.tensor(idx, dtype=torch.long, device="cpu")


def cvarr_losses(losses: torch.Tensor, rho: float) -> torch.Tensor:
    """Average the largest ``rho`` fraction of per-donor losses (CVaR)."""
    if losses.numel() == 0:
        return torch.zeros((), device=losses.device)
    k = max(1, int(np.ceil(rho * losses.numel())))
    top = torch.topk(losses, k).values
    return top.mean()


def _posterior(features: torch.Tensor, T: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(tau * features @ T.t(), dim=-1)


def compute_margin(posterior: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(posterior, 2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def donor_valid(
    r_i: torch.Tensor,        # (n_i, d) residuals of the query domain
    s_i: torch.Tensor,        # (n_i, d) semantic components of the query domain
    y_i: torch.Tensor,        # (n_i,) classes (ground truth for source, frozen-text for target)
    r_j: torch.Tensor,        # (n_j, d) residuals of the opposite domain
    y_j: torch.Tensor,        # (n_j,) classes of the donors
    b_j: torch.Tensor,        # (n_j,) donor energy bins
    b_i: torch.Tensor,        # (n_i,) query energy bins
    m_j: torch.Tensor,        # (n_j,) donor text margins (0 for source donors)
    donor_is_target: bool,    # whether donors come from the target domain
    kappa_donor: float,
    kappa_r: float,
    j_min: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (valid_mask (n_i,), donor_count (n_i,)).

    A query is donor-valid if it has at least ``j_min`` admissible donors
    satisfying same-class, same-bin, energy-ratio and (target-only) margin
    constraints. Donors are the opposite domain (the caller selects them).
    """
    n_i = r_i.shape[0]
    if r_j.shape[0] == 0:
        return torch.zeros(n_i, dtype=torch.bool, device=r_i.device), torch.zeros(n_i, dtype=torch.long, device=r_i.device)
    s_norm = s_i.norm(dim=1).clamp(min=eps)          # (n_i,)
    r_norm_i = r_i.norm(dim=1)                        # (n_i,)
    r_norm_j = r_j.norm(dim=1)                        # (n_j,)
    valid = torch.zeros(n_i, dtype=torch.bool, device=r_i.device)
    counts = torch.zeros(n_i, dtype=torch.long, device=r_i.device)
    for i in range(n_i):
        same_cls = (y_j == y_i[i])
        if donor_is_target:
            same_cls = same_cls & (m_j >= kappa_donor)
        same_bin = (b_j == b_i[i])
        ratio = r_norm_j / (s_norm[i] + eps)
        energy_ok = ratio <= kappa_r
        mask = same_cls & same_bin & energy_ok
        cnt = int(mask.sum())
        counts[i] = cnt
        valid[i] = cnt >= j_min
    return valid, counts


def transport_features(s: torch.Tensor, r_j: torch.Tensor) -> torch.Tensor:
    """z~_{i<-j} = norm(s_i + r_j) for one query and its donor set."""
    z = s.unsqueeze(0) + r_j.unsqueeze(1)  # (n_j, 1, d) + (1, n_q, d)? -> handle below
    return F.normalize(z, dim=-1)


def per_query_transport(s_i: torch.Tensor, r_j: torch.Tensor) -> torch.Tensor:
    """For one query semantic component and a donor residual set:
    returns (n_j, d) normalized transported features."""
    z = s_i.unsqueeze(0) + r_j   # (n_j, d)
    return F.normalize(z, dim=-1)


def compute_certificate_batch(
    z_q: torch.Tensor,            # (n, d) query features (normalized)
    s_q: torch.Tensor,            # (n, d) query semantic components
    r_q: torch.Tensor,            # (n, d) query residuals
    y_q: torch.Tensor,            # (n,) query classes
    T: torch.Tensor,              # (K, d) frozen text prototypes
    tau: float,
    r_donor: torch.Tensor,        # (n_d, d) opposite-domain donor residuals
    y_donor: torch.Tensor,        # (n_d,)
    b_donor: torch.Tensor,        # (n_d,)
    b_query: torch.Tensor,        # (n,)
    m_donor: torch.Tensor,        # (n_d,)
    donor_is_target: bool,
    kappa_donor: float,
    kappa_r: float,
    j_min: int,
    rho: float,
    eta: float,
    kappa: float,
    zeta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute u_i, c_i, valid mask, and donor counts for a query batch.

    Samples without enough valid donors get c_i = 0 and are excluded (the
    conservative missing-data rule).
    """
    n = z_q.shape[0]
    device = z_q.device
    P_q = _posterior(z_q, T, tau)                       # (n, K)
    m_q = compute_margin(P_q)                           # (n,)
    valid, counts = donor_valid(
        r_q, s_q, y_q, r_donor, y_donor, b_donor, b_query,
        m_donor, donor_is_target, kappa_donor, kappa_r, j_min,
    )
    u = torch.zeros(n, device=device)
    for i in range(n):
        if not valid[i]:
            continue
        same = (y_donor == y_q[i])
        if donor_is_target:
            same = same & (m_donor >= kappa_donor)
        same_bin = (b_donor == b_query[i])
        s_norm = s_q[i].norm().clamp(min=1e-8)
        ratio = r_donor.norm(dim=1) / s_norm
        mask = same & same_bin & (ratio <= kappa_r)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            valid[i] = False
            continue
        z_t = per_query_transport(s_q[i], r_donor[idx])  # (n_j, d)
        P_t = _posterior(z_t, T, tau)                    # (n_j, K)
        q = P_q[i]
        kl = (q * torch.log(q.clamp(min=1e-12) / P_t.clamp(min=1e-12))).sum(dim=-1)  # (n_j,)
        u[i] = cvarr_losses(kl, rho)
    c = torch.zeros(n, device=device)
    c[valid] = torch.exp(-u[valid] / eta) * torch.sigmoid((m_q[valid] - kappa) / zeta)
    return u, c, valid, counts


def select_certified_target(
    c: torch.Tensor,     # (n_t,)
    y_t: torch.Tensor,   # (n_t,) frozen-text pseudo-labels
    valid_t: torch.Tensor,
    pi_e: float,
) -> torch.Tensor:
    """Class-wise top-pi_e certified set I_t (by c descending)."""
    n = c.shape[0]
    sel = torch.zeros(n, dtype=torch.bool, device=c.device)
    cand = valid_t.nonzero(as_tuple=True)[0]
    if cand.numel() == 0:
        return sel
    classes = torch.unique(y_t[cand])
    for cls in classes.tolist():
        idx = cand[y_t[cand] == cls]
        vals = c[idx]
        k = max(1, int(np.ceil(pi_e * idx.numel())))
        k = min(k, idx.numel())
        if k <= 0:
            continue
        top = torch.topk(vals, k).indices
        sel[idx[top]] = True
    return sel
