"""Loss terms of the hierarchical objective (Eq. total_objective).

Each function returns the raw (unweighted) term; the trainer applies the
config weights. The gradient-reversal layer implements the L_leak direction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, gradient reversal with scale ``lambda`` backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, scale)


def text_subspace_loss(V: torch.Tensor, V_T: torch.Tensor) -> torch.Tensor:
    """L_text = ||V V^T - V_T V_T^T||_F^2 (Eq. text_subspace)."""
    return (V @ V.t() - V_T @ V_T.t()).pow(2).sum()


def factorization_loss(
    Z: torch.Tensor,        # (n, d)
    Pbar: torch.Tensor,     # (n, K)
    C: torch.Tensor,        # (K, r)
    V: torch.Tensor,        # (d, r)
    A: torch.Tensor,        # (n, m)
    B: torch.Tensor,        # (d, m)
    E: torch.Tensor,        # (n, d)
    beta: float,
    xi: float,
    gamma: float,
) -> Tuple[torch.Tensor, dict]:
    """L_fac for one domain (Eq. factorization_loss), without the alpha L_text
    term which is added by the trainer with the global text-subspace penalty.
    Reconstruction uses the mean over entries for numerical stability."""
    recon = Z - Pbar @ C @ V.t() - A @ B.t() - E
    l_rec = 0.5 * recon.pow(2).mean()
    l_b = beta * (A.pow(2).sum() + B.pow(2).sum()) / Z.shape[0]
    l_out = xi * E.norm(p=2, dim=1).sum() / Z.shape[0]   # ||E||_{2,1}
    l_orth = gamma * ((V.t() @ B).pow(2).sum()) / (V.shape[1] * B.shape[1])
    loss = l_rec + l_b + l_out + l_orth
    return loss, {"l_rec": l_rec.item(), "l_b": l_b.item(),
                  "l_out": l_out.item(), "l_orth": l_orth.item()}


def domain_loss(D_dom_logits_s: torch.Tensor, D_dom_logits_t: torch.Tensor) -> torch.Tensor:
    """L_dom = CE(D_dom(r), domain) over source + target (Eq. domain_loss)."""
    logits = torch.cat([D_dom_logits_s, D_dom_logits_t], dim=0)
    labels = torch.cat([
        torch.zeros(D_dom_logits_s.shape[0], dtype=torch.long, device=logits.device),
        torch.ones(D_dom_logits_t.shape[0], dtype=torch.long, device=logits.device),
    ])
    return F.cross_entropy(logits, labels)


def leakage_loss(D_cls_logits: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
    """L_leak = CE(D_cls(GRL(r)), y_src) (Eq. leakage_loss)."""
    return F.cross_entropy(D_cls_logits, y_src)


def clip_alignment_loss(
    qT_s: torch.Tensor,     # (n_s, K) frozen text posterior of source
    y_s: torch.Tensor,      # (n_s,)
    qT_t: torch.Tensor,     # (n_t, K)
    lambda_ent: float,
) -> Tuple[torch.Tensor, dict]:
    """L_clip = source CE + entropy minimization + IM (Eq. clip_alignment)."""
    l_src = F.cross_entropy(qT_s, y_s)
    p_mean = qT_t.mean(dim=0)
    l_ent = -qT_t * torch.log(qT_t.clamp(min=1e-12))
    l_ent = l_ent.sum(dim=1).mean() - (-p_mean * torch.log(p_mean.clamp(min=1e-12))).sum()
    loss = l_src + lambda_ent * l_ent
    return loss, {"l_src": l_src.item(), "l_ent": l_ent.item()}


def source_counterfactual_loss(
    qT_transports: list,    # list over queries of (n_j, K) transported posteriors
    y_src: torch.Tensor,    # (n_vs,) source labels of valid queries
) -> torch.Tensor:
    """L_cf-src (Eq. source_counterfactual): mean CE over donors per query."""
    if len(qT_transports) == 0:
        return torch.zeros((), device=y_src.device)
    total = torch.zeros((), device=y_src.device)
    n_q = 0
    for k, P in enumerate(qT_transports):
        target = y_src[k]
        total = total + F.cross_entropy(P, target.expand(P.shape[0]))
        n_q += 1
    return total / max(n_q, 1)


def certificate_loss(u: torch.Tensor, valid: torch.Tensor,
                     l_cf_src: torch.Tensor, lambda_cf: float) -> Tuple[torch.Tensor, dict]:
    """L_cert = mean(u over V) + lambda_cf L_cf-src (Eq. certificate_loss)."""
    if valid.sum() == 0:
        return torch.zeros((), device=u.device), {"l_cert": 0.0}
    l_u = u[valid].mean()
    loss = l_u + lambda_cf * l_cf_src
    return loss, {"l_cert": loss.item(), "l_u": l_u.item(), "l_cf": l_cf_src.item()}


def supervised_loss(
    p_s: torch.Tensor, y_s: torch.Tensor,
    p_t_cert: torch.Tensor, c_t_cert: torch.Tensor, y_t_cert: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """L_sup (Eq. supervised_loss):
    (1/N_s) sum CE(p_i, y_i) + (1/max(|I_t|,1)) sum sg(c_i) CE(p_i, yhat_i).
    """
    l_src = F.cross_entropy(p_s, y_s)
    n_cert = p_t_cert.shape[0]
    if n_cert == 0:
        loss = l_src
        l_tgt = torch.zeros((), device=p_s.device)
    else:
        w = c_t_cert.detach()  # sg(c_i)
        l_tgt = (w * F.cross_entropy(p_t_cert, y_t_cert, reduction="none")).sum() / n_cert
        loss = l_src + l_tgt
    return loss, {"l_sup_src": l_src.item(), "l_sup_tgt": float(l_tgt.item())}


def total_loss(terms: dict, weights: dict) -> torch.Tensor:
    """Assemble the three functional layers (Eq. total_objective)."""
    loss = torch.zeros((), device=next(iter(terms.values())).device)
    for key, w in weights.items():
        if w is None or w == 0.0:
            continue
        t = terms.get(key)
        if t is not None and t.requires_grad:
            loss = loss + w * t
    return loss
