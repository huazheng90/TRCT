"""TCRT learnable modules.

- ``TextAnchoredFactorization``: the shared-private decomposition. The semantic
  basis ``V`` is the thin-Q factor of an unconstrained ``Vbar`` (Stiefel
  implementation), the private residual bases ``B_s/B_t`` are rank-``m``
  global parameters, and the batch-local coefficient matrices ``A_d`` and
  outlier matrices ``E_d`` are solved by the alternating procedure in
  :mod:`src.factorization`.
- ``DomainDiscriminator`` / ``ClassDiscriminator``: functional disentanglement
  heads (L_dom and L_leak in the paper).
- ``AnchorBank``: the K x M_c semantic anchors with k-means initialization and
  EMA updates (Eq. anchor_update).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def thin_qf(Vbar: torch.Tensor) -> torch.Tensor:
    """Differentiable retraction of ``Vbar`` onto the Stiefel manifold.

    Returns the thin-Q factor of the QR decomposition with the sign of the
    diagonal of R fixed positive, so that the parametrization is unique.
    """
    Q, R = torch.linalg.qr(Vbar, mode="reduced")
    sign = torch.sign(torch.diag(R))
    # Guard against zero diagonal (degenerate column): treat as +1.
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(0)
    return Q


class TextAnchoredFactorization(nn.Module):
    """Learns V (text-anchored semantic subspace) and B_s/B_t (private bases)."""

    def __init__(self, d: int, r: int, m: int, init_V: Optional[torch.Tensor] = None,
                 perturb: float = 1e-3):
        super().__init__()
        self.d = d
        self.r = r
        self.m = m
        if init_V is not None:
            Vbar0 = init_V.detach().clone()
        else:
            Vbar0 = torch.eye(d, r)
        if perturb > 0:
            Vbar0 = Vbar0 + perturb * torch.randn_like(Vbar0)
        self.Vbar = nn.Parameter(Vbar0)
        self.B_s = nn.Parameter(torch.randn(d, m) * 0.02)
        self.B_t = nn.Parameter(torch.randn(d, m) * 0.02)

    def V(self) -> torch.Tensor:
        return thin_qf(self.Vbar)

    def residual(self, A: torch.Tensor, domain: str) -> torch.Tensor:
        B = self.B_s if domain == "s" else self.B_t
        return A @ B.t()

    def orthogonality_penalty(self) -> torch.Tensor:
        """sum_d ||V^T B_d||_F^2 (gamma term in L_fac)."""
        V = self.V()
        return F.mse_loss(V.t() @ self.B_s, torch.zeros(self.r, self.m, device=V.device)) + \
               F.mse_loss(V.t() @ self.B_t, torch.zeros(self.r, self.m, device=V.device))


class _MLPDiscriminator(nn.Module):
    def __init__(self, d: int, out: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainDiscriminator(_MLPDiscriminator):
    """Binary domain head D_dom on the private residual."""

    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__(d, 2, hidden, dropout)


class ClassDiscriminator(_MLPDiscriminator):
    """Source-class head D_cls on the (gradient-reversed) residual."""

    def __init__(self, d: int, num_classes: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__(d, num_classes, hidden, dropout)


class AnchorBank(nn.Module):
    """K x M_c semantic anchors in the shared subspace (dimension r)."""

    def __init__(self, K: int, M_c: int, r: int, mu: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.K = K
        self.M_c = M_c
        self.r = r
        self.mu = mu
        self.eps = eps
        # anchors[c, ell] in R^r, initialized by k-means on labeled source h.
        self.register_buffer("anchors", torch.zeros(K, M_c, r))
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize_from_source(self, h_s: torch.Tensor, y_s: torch.Tensor) -> None:
        """Class-wise k-means over labeled source semantic coordinates."""
        K, M_c = self.K, self.M_c
        for c in range(K):
            idx = (y_s == c)
            if idx.sum() == 0:
                continue
            feats = h_s[idx]  # (n_c, r)
            n = feats.shape[0]
            if n <= M_c:
                self.anchors[c, :n] = F.normalize(feats[:n], dim=-1)
                for ell in range(n, M_c):
                    self.anchors[c, ell] = self.anchors[c, ell - 1]
                continue
            # k-means with a few restarts, deterministic seed.
            best = None
            best_obj = float("inf")
            for _ in range(5):
                perm = torch.randperm(n, device=feats.device)[:M_c]
                centroids = F.normalize(feats[perm], dim=-1).clone()
                for _ in range(20):
                    sim = feats @ centroids.t()  # (n, M_c)
                    assign = sim.argmax(dim=1)
                    new_cent = []
                    for ell in range(M_c):
                        grp = feats[assign == ell]
                        if grp.shape[0] > 0:
                            new_cent.append(F.normalize(grp.mean(dim=0), dim=-1))
                        else:
                            new_cent.append(centroids[ell])
                    centroids = torch.stack(new_cent)
                sim = feats @ centroids.t()
                obj = float((sim.max(dim=1).values * -1).sum().item())
                if obj < best_obj:
                    best_obj = obj
                    best = centroids
            self.anchors[c] = best
        self._initialized.fill_(True)

    def assign(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Nearest anchor per sample within its (predicted/known) class."""
        anchors = F.normalize(self.anchors, dim=-1)
        h_n = F.normalize(h, dim=-1)
        out = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        for i in range(h.shape[0]):
            c = int(y[i])
            sim = h_n[i] @ anchors[c].t()  # (M_c,)
            out[i] = c * self.M_c + int(sim.argmax())
        return out

    @torch.no_grad()
    def ema_update(self, h: torch.Tensor, w: torch.Tensor, assign: torch.Tensor) -> None:
        """One anchor update per iteration (Eq. anchor_update)."""
        if h.shape[0] == 0:
            return
        anchors = self.anchors.clone()
        h_n = F.normalize(h, dim=-1)
        for i in range(h.shape[0]):
            m = int(assign[i])
            c = m // self.M_c
            ell = m % self.M_c
            num = anchors[c, ell] * self.mu + (1 - self.mu) * w[i] * h_n[i]
            den = self.mu + (1 - self.mu) * w[i] + self.eps
            anchors[c, ell] = num / den
        self.anchors.copy_(F.normalize(anchors, dim=-1))

    def affinity(self, h: torch.Tensor, y: torch.Tensor, tau_g: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class-masked sample-anchor affinities Q (Eq. anchor_affinity).

        Returns (Q, anchor_ids) where anchor_ids[m] is the global anchor index
        of column m; inactive (zero-degree) anchors are removed by the caller.
        """
        h_n = F.normalize(h, dim=-1)
        anchors = F.normalize(self.anchors, dim=-1)  # (K, M_c, r)
        n = h.shape[0]
        rows = []
        for i in range(n):
            c = int(y[i])
            sims = h_n[i] @ anchors[c].t()  # (M_c,)
            logits = tau_g * sims
            probs = torch.softmax(logits, dim=-1)
            rows.append(probs)
        Q = torch.stack(rows)  # (n, M_c) with per-class softmax
        anchor_ids = torch.arange(self.K * self.M_c, device=h.device).view(self.K, self.M_c)
        anchor_ids = anchor_ids[y]  # (n, M_c) global ids of compatible anchors
        return Q, anchor_ids
