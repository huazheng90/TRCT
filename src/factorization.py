"""Batch-local alternating solver for the decomposition (Algorithm 1, inner).

For each paired mini-batch we alternate ``J_alt`` block updates:

1. ``A_d``: one gradient step on the batch factorization objective plus the
   functional-disentanglement terms (residual bases and discriminators are
   detached — no gradient is unrolled through this local solver).
2. ``E_d``: the closed-form row-wise proximal update (Eq. proximal_outlier).

The returned ``(A, E)`` are detached and held fixed for the global backward
pass, exactly as in the paper: "no gradient is unrolled through the local
alternating solver; the global gradient is evaluated at its current block
solution."
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.model import ClassDiscriminator, DomainDiscriminator


def _proximal_rows(U: torch.Tensor, xi: float, eps: float) -> torch.Tensor:
    norms = U.norm(dim=1, keepdim=True)
    scale = torch.clamp(1.0 - xi / torch.clamp(norms, min=eps), min=0.0)
    return scale * U


def solve_decomposition(
    Z: torch.Tensor,
    Pbar: torch.Tensor,
    C: torch.Tensor,
    V: torch.Tensor,
    B: torch.Tensor,
    dom_labels: torch.Tensor,       # (n,) 0=source, 1=target
    y_src: Optional[torch.Tensor],  # (n,) source labels (None for target)
    D_dom: Optional[DomainDiscriminator],
    D_cls: Optional[ClassDiscriminator],
    lambda_priv: float,
    lambda_leak: float,
    beta: float,
    xi: float,
    eps_prox: float,
    J_alt: int,
    lr_block: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alternating solve for one domain's (A, E).

    The A-update minimizes
        0.5||Z - Pbar C V^T - A B^T - E||^2 + beta||A||^2
        + lambda_priv * ( CE(D_dom(AB^T), dom) - lambda_leak * CE(D_cls(AB^T), y_src) )
    where the L_leak gradient on the residual coefficients is reversed (GRL).
    ``D_dom``/``D_cls`` are used detached (their own parameters are updated
    during the global backward pass). ``y_src`` is provided only for the
    source domain.
    """
    n = Z.shape[0]
    device = Z.device
    A = torch.zeros(n, B.shape[1], device=device)
    E = torch.zeros_like(Z)
    CVt = C @ V.t()

    def _block_loss(Av: torch.Tensor) -> torch.Tensor:
        r = Av @ B.t()
        recon = Z - Pbar @ CVt - r - E
        loss = 0.5 * recon.pow(2).mean() + beta * Av.pow(2).mean()
        if D_dom is not None:
            loss = loss + lambda_priv * F.cross_entropy(D_dom(r.detach()), dom_labels)
        if y_src is not None and D_cls is not None:
            leak = F.cross_entropy(D_cls(r.detach()), y_src)
            loss = loss - lambda_priv * lambda_leak * leak
        return loss

    for _ in range(J_alt):
        with torch.enable_grad():
            A.requires_grad_(True)
            loss = _block_loss(A)
            grad = torch.autograd.grad(loss, A)[0]
        with torch.no_grad():
            A = A - lr_block * grad
        A = A.detach()
        U = Z - Pbar @ CVt - A @ B.t()
        E = _proximal_rows(U, xi, eps_prox)

    return A.detach(), E.detach()
