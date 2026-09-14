"""TCRT trainer: warm-up then certificate-guided adaptation.

Implements Algorithm 1. The vision tower is frozen and image features are
cached ahead of time (see ``data.py``), so each iteration operates on
``(z_s, y_s, z_t)`` feature pairs. The frozen text prototypes ``T`` are
recomputed every iteration because the prompt context vectors are trainable.

Warm-up stage (E_w epochs): source supervision + prompt alignment +
factorization + functional disentanglement only.
Adaptation stage: certificates, anchors and the certified graph are updated
once per iteration; the hierarchical objective is back-propagated to prompts,
``Vbar``, ``B_s/B_t`` and the discriminators.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import losses as L
from src.certificate import (EnergyBins, ResidualQueue, compute_certificate_batch,
                             compute_margin, select_certified_target)
from src.factorization import solve_decomposition
from src.graph import build_certified_graph, spectral_tail_loss
from src.model import AnchorBank, ClassDiscriminator, DomainDiscriminator, TextAnchoredFactorization
from src.utils import LOGGER


class FeatureSampler:
    """Shuffle-once epoch sampler over cached feature arrays."""

    def __init__(self, feats: torch.Tensor, labels: Optional[torch.Tensor],
                 batch_size: int, iters_per_epoch: int, seed: int):
        self.feats = feats
        self.labels = labels
        self.batch_size = batch_size
        self.iters = iters_per_epoch
        self.g = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(feats.shape[0], generator=self.g)
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n = self.feats.shape[0]
        idx = []
        for _ in range(self.batch_size):
            if self.pos >= n:
                self.order = torch.randperm(n, generator=self.g)
                self.pos = 0
            idx.append(int(self.order[self.pos]))
            self.pos += 1
        idx = torch.tensor(idx, device=self.feats.device)
        z = self.feats[idx]
        y = self.labels[idx] if self.labels is not None else None
        return z, y


class TCRTTrainer:
    def __init__(self, cfg: dict, backend: nn.Module, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.backend = backend
        d = backend.d
        self.K = cfg["num_classes"]
        self.d = d

        r = self._text_rank(cfg)
        m = cfg["residual_rank"]
        # V_T = rank-r right singular subspace of T (computed lazily in fit).
        self.r = r
        self.factorization = TextAnchoredFactorization(
            d=d, r=r, m=m, perturb=cfg.get("init_perturb", 1e-3)
        ).to(device)
        self.D_dom = DomainDiscriminator(d, hidden=cfg.get("disc_hidden", 256),
                                         dropout=cfg.get("disc_dropout", 0.0)).to(device)
        self.D_cls = ClassDiscriminator(d, self.K, hidden=cfg.get("disc_hidden", 256),
                                        dropout=cfg.get("disc_dropout", 0.0)).to(device)
        self.anchors = AnchorBank(self.K, cfg["anchors_per_class"],
                                  r, mu=cfg.get("anchor_momentum", 0.99)).to(device)
        self.energy_bins = EnergyBins(cfg.get("energy_bins", 10))
        self.queues = {"s": ResidualQueue(cfg.get("queue_capacity", 4096)),
                       "t": ResidualQueue(cfg.get("queue_capacity", 4096))}
        self.V_T: Optional[torch.Tensor] = None
        self.text_logit_scale = float(getattr(backend, "text_logit_scale", 100.0))
        self.cfg_scale = float(cfg.get("text_logit_scale", 0.0))
        if self.cfg_scale > 0:
            self.text_logit_scale = self.cfg_scale

        self.opt_params = self._collect_trainable()
        self.optimizer = torch.optim.SGD(
            self.opt_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            momentum=cfg.get("momentum", 0.9),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["warmup_epochs"] + cfg["adapt_epochs"], eta_min=1e-6
        )
        self.step = 0

    # ------------------------------------------------------------------ utils
    def _text_rank(self, cfg: dict) -> int:
        r0 = cfg.get("semantic_rank", 64)
        eps = cfg.get("rank_eps", 1e-3)
        # Determined from T at fit time; fall back to r0 for the mock path.
        return r0

    def _collect_trainable(self) -> List[torch.nn.Parameter]:
        params = []
        for mod in [self.backend, self.factorization, self.D_dom, self.D_cls]:
            for p in mod.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def _text_prototypes(self, class_names: Sequence[str]) -> torch.Tensor:
        T = self.backend.text_prototypes(class_names)
        return F.normalize(T, dim=-1)

    def _posterior(self, z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.text_logit_scale * z @ T.t(), dim=-1)

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "factorization": self.factorization.state_dict(),
            "D_dom": self.D_dom.state_dict(),
            "D_cls": self.D_cls.state_dict(),
            "anchors": self.anchors.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "V_T": self.V_T,
            "cfg": self.cfg,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        LOGGER.info("checkpoint saved to %s", path)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.factorization.load_state_dict(state["factorization"])
        self.D_dom.load_state_dict(state["D_dom"])
        self.D_cls.load_state_dict(state["D_cls"])
        self.anchors.load_state_dict(state["anchors"])
        self.V_T = state.get("V_T")
        self.step = state.get("step", 0)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

    # ------------------------------------------------------------- features
    def _project(self, z: torch.Tensor, T: torch.Tensor,
                 y: Optional[torch.Tensor], domain: str):
        """Decompose one mini-batch: returns a dict of tensors.

        With ``use_factorization=false`` (the controlled prompt-UDA base,
        Table system_ablation first row) the decomposition is bypassed: the
        classifier is the frozen text posterior and residuals are zero.
        """
        q = self._posterior(z, T)
        if not self.cfg.get("use_factorization", True):
            n = z.shape[0]
            Pbar = F.one_hot(y, num_classes=self.K).float() \
                if (domain == "s" and y is not None) else q.detach()
            return {"z": z, "h": z, "p": q, "q": q, "s": z,
                    "r": torch.zeros_like(z),
                    "A": torch.zeros(n, self.cfg["residual_rank"], device=z.device),
                    "E": torch.zeros_like(z), "Pbar": Pbar, "C": T, "V": None}
        V = self.factorization.V()
        C = T @ V                                   # (K, r)
        h = z @ V                                  # (n, r)
        p = torch.softmax(self.text_logit_scale * h @ C.t(), dim=-1)  # adaptive posterior
        if domain == "s" and y is not None:
            Pbar = F.one_hot(y, num_classes=self.K).float()
        else:
            Pbar = p.detach()
        B = self.factorization.B_s if domain == "s" else self.factorization.B_t
        dom_labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device) if domain == "s" \
            else torch.ones(z.shape[0], dtype=torch.long, device=z.device)
        A, E = solve_decomposition(
            Z=z, Pbar=Pbar, C=C, V=V, B=B, dom_labels=dom_labels,
            y_src=y if domain == "s" else None,
            D_dom=self.D_dom if self.cfg.get("use_disentangle", True) else None,
            D_cls=self.D_cls if (self.cfg.get("use_disentangle", True) and domain == "s") else None,
            lambda_priv=self.cfg["lambda_priv"],
            lambda_leak=self.cfg.get("lambda_leak", 0.1),
            beta=self.cfg.get("beta", 1e-3),
            xi=self.cfg.get("xi", 1e-3),
            eps_prox=self.cfg.get("eps_prox", 1e-8),
            J_alt=self.cfg.get("j_alt", 5),
            lr_block=self.cfg.get("lr_block", 0.1),
        )
        s = Pbar @ C @ V.t()
        r = A @ B.t()
        return {"z": z, "h": h, "p": p, "q": q, "s": s, "r": r, "A": A, "E": E,
                "Pbar": Pbar, "C": C, "V": V}

    # -------------------------------------------------------------- losses
    def _warmup_loss(self, fs: dict, ft: dict, T: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        l_clip, m_clip = L.clip_alignment_loss(fs["q"], fs["y"], ft["q"], cfg["lambda_ent"])
        l_fac = torch.zeros((), device=fs["z"].device)
        if cfg.get("use_factorization", True):
            l_fac_s, m_s = L.factorization_loss(fs["z"], fs["Pbar"], fs["C"], fs["V"],
                                                fs["A"], self.factorization.B_s, fs["E"],
                                                cfg.get("beta", 1e-3), cfg.get("xi", 1e-3),
                                                cfg.get("gamma", 1e-2))
            l_fac_t, m_t = L.factorization_loss(ft["z"], ft["Pbar"], ft["C"], ft["V"],
                                                ft["A"], self.factorization.B_t, ft["E"],
                                                cfg.get("beta", 1e-3), cfg.get("xi", 1e-3),
                                                cfg.get("gamma", 1e-2))
            l_text = L.text_subspace_loss(fs["V"], self.V_T)
            l_fac = l_fac_s + l_fac_t + cfg.get("alpha", 1.0) * l_text

        l_dom = torch.zeros((), device=fs["z"].device)
        l_leak = torch.zeros((), device=fs["z"].device)
        if cfg.get("use_factorization", True) and cfg.get("use_disentangle", True):
            l_dom = L.domain_loss(self.D_dom(fs["r"]), self.D_dom(ft["r"]))
            l_leak = L.leakage_loss(self.D_cls(fs["r"]), fs["y"])

        loss = (l_clip + cfg["lambda_fac"] * l_fac
                + cfg["lambda_priv"] * (l_dom + cfg.get("lambda_leak", 0.1) * l_leak))
        return loss, {"l_clip": l_clip.item(), "l_fac": l_fac.item(),
                      "l_dom": l_dom.item(), "l_leak": l_leak.item()}

    def _adaptation_loss(self, fs: dict, ft: dict, T: torch.Tensor,
                         pi_e: float, epoch: int) -> Tuple[torch.Tensor, dict, dict]:
        cfg = self.cfg
        device = fs["z"].device
        yhat_t = ft["q"].argmax(dim=1)  # frozen-text pseudo-labels

        # --- donor pools: opposite-domain current batch + queue -------------
        def _c0(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if a.shape[0] == 0:
                return b
            if b.shape[0] == 0:
                return a
            return torch.cat([a, b], dim=0)

        # Source queries <- target donors; target queries <- source donors.
        q_s_r, q_s_c, q_s_b, q_s_m = self.queues["s"].as_tensors(device)
        q_t_r, q_t_c, q_t_b, q_t_m = self.queues["t"].as_tensors(device)

        # Current-batch donor residuals with metadata.
        cur_s_b = self.energy_bins.assign(fs["r"].norm(dim=1)).to(device)
        cur_t_b = self.energy_bins.assign(ft["r"].norm(dim=1)).to(device)
        # Source donors have no margin constraint; store zeros.
        cur_s_m = torch.zeros(fs["z"].shape[0], device=device)
        cur_t_m = compute_margin(ft["q"])

        # Target donors = current target batch + target queue.
        r_donor_t = _c0(ft["r"], q_t_r)
        y_donor_t = _c0(yhat_t, q_t_c)
        b_donor_t = _c0(cur_t_b, q_t_b)
        m_donor_t = _c0(cur_t_m, q_t_m)

        # Source donors = current source batch + source queue.
        r_donor_s = _c0(fs["r"], q_s_r)
        y_donor_s = _c0(fs["y"], q_s_c)
        b_donor_s = _c0(cur_s_b, q_s_b)
        m_donor_s = _c0(torch.zeros(fs["z"].shape[0], device=device), q_s_m)

        # --- certificate for source queries ---------------------------------
        u_s, c_s, valid_s, cnt_s = compute_certificate_batch(
            fs["z"], fs["s"], fs["r"], fs["y"], T, self.text_logit_scale,
            r_donor_t, y_donor_t, b_donor_t, cur_s_b, m_donor_t,
            donor_is_target=True,
            kappa_donor=cfg.get("kappa_donor", 0.05),
            kappa_r=cfg.get("kappa_r", 2.0),
            j_min=cfg.get("j_min", 1),
            rho=cfg.get("rho", 0.1), eta=cfg.get("eta", 0.1),
            kappa=cfg.get("kappa_cert", 0.0), zeta=cfg.get("zeta", 0.1),
        )
        u_t, c_t, valid_t, cnt_t = compute_certificate_batch(
            ft["z"], ft["s"], ft["r"], yhat_t, T, self.text_logit_scale,
            r_donor_s, y_donor_s, b_donor_s, cur_t_b, m_donor_s,
            donor_is_target=False,
            kappa_donor=cfg.get("kappa_donor", 0.05),
            kappa_r=cfg.get("kappa_r", 2.0),
            j_min=cfg.get("j_min", 1),
            rho=cfg.get("rho", 0.1), eta=cfg.get("eta", 0.1),
            kappa=cfg.get("kappa_cert", 0.0), zeta=cfg.get("zeta", 0.1),
        )
        # --- certified target set --------------------------------------------
        # ``selection`` controls the pseudo-label filter (Table system_ablation):
        #   certificate  -> donor-validated targets, class-wise top pi_e by c_i
        #   confidence   -> class-wise top pi_e by frozen-text confidence
        #   random       -> class-wise random pi_e
        #   none         -> no target supervision / no graph
        if not cfg.get("use_certificate", True):
            I_t = self._select_targets(ft, None, yhat_t, None, pi_e)
        else:
            I_t = self._select_targets(ft, c_t, yhat_t, valid_t, pi_e)

        # --- certificate loss ------------------------------------------------
        l_cert = torch.zeros((), device=device)
        if cfg.get("use_certificate", True):
            l_cf_src = torch.zeros((), device=device)
            if cfg.get("use_counterfactual", True) and valid_s.sum() > 0:
                transports = []
                for i in range(fs["z"].shape[0]):
                    if not valid_s[i]:
                        continue
                    same = (y_donor_t == fs["y"][i]) & (m_donor_t >= cfg.get("kappa_donor", 0.05))
                    same_bin = (b_donor_t == cur_s_b[i])
                    s_n = fs["s"][i].norm().clamp(min=1e-8)
                    ratio = r_donor_t.norm(dim=1) / s_n
                    idx = (same & same_bin & (ratio <= cfg.get("kappa_r", 2.0))).nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    z_t = F.normalize(fs["s"][i].unsqueeze(0) + r_donor_t[idx], dim=-1)
                    transports.append(self._posterior(z_t, T))
                if transports:
                    l_cf_src = L.source_counterfactual_loss(transports, fs["y"][valid_s][:len(transports)])
            l_cert, m_cert = L.certificate_loss(
                torch.cat([u_s, u_t]), torch.cat([valid_s, valid_t]),
                l_cf_src, cfg.get("lambda_cf", 1.0))

        # --- anchors + graph -------------------------------------------------
        l_tail = torch.zeros((), device=device)
        use_graph = cfg.get("use_graph", True)
        if I_t.sum() > 0:
            w_anchor = torch.cat([
                torch.ones(fs["z"].shape[0], device=device),   # source w=1
                c_t[I_t].detach(),                              # certified target w=sg(c)
            ])
            h_anchor = torch.cat([fs["h"], ft["h"][I_t]], dim=0)
            y_anchor = torch.cat([fs["y"], yhat_t[I_t]], dim=0)
            assign = self.anchors.assign(h_anchor, y_anchor)
            self.anchors.ema_update(h_anchor, w_anchor, assign)

            if use_graph:
                Q, inv_sqrt_dx, inv_da, active_ids = build_certified_graph(
                    ft["h"][I_t], yhat_t[I_t], self.anchors,
                    cfg.get("tau_g", 10.0), eps_deg=cfg.get("eps_deg", 1e-6))
                # Graph-edge corruption wrapper (Table edge_corruption): rewire a
                # fraction of sample-anchor pairs to a same-class anchor, keeping
                # the row weight (sample degree) and endpoint type; no labels.
                if Q is not None and cfg.get("corrupt_edges", 0.0) > 0:
                    Q = self._corrupt_edges(Q, active_ids, cfg["corrupt_edges"],
                                            cfg.get("corrupt_seed", 0) + self.step)
                if Q is not None:
                    l_tail = spectral_tail_loss(ft["p"][I_t], Q, inv_sqrt_dx, inv_da,
                                                p=cfg.get("spectral_p", 1))
        else:
            self.anchors.ema_update(fs["h"], torch.ones(fs["z"].shape[0], device=device),
                                    self.anchors.assign(fs["h"], fs["y"]))

        # --- supervision ------------------------------------------------------
        l_sup, m_sup = L.supervised_loss(
            fs["p"], fs["y"],
            ft["p"][I_t], c_t[I_t].detach(), yhat_t[I_t])

        # --- pseudo-label corruption wrapper (Table label_corruption) ---------
        if cfg.get("corrupt_labels", 0.0) > 0 and I_t.sum() > 0:
            q = cfg["corrupt_labels"]
            rng = np.random.default_rng(cfg.get("corrupt_seed", 0) + self.step)
            n_c = int(I_t.sum())
            n_swap = int(round(q * n_c))
            if n_swap > 0:
                swap_idx = rng.choice(n_c, size=n_swap, replace=False)
                bad = rng.integers(0, self.K - 1, size=n_swap)  # wrong class
                new_y = yhat_t[I_t].clone()
                for k in range(n_swap):
                    new_y[swap_idx[k]] = (new_y[swap_idx[k]] + 1 + bad[k]) % self.K
                l_sup, m_sup = L.supervised_loss(fs["p"], fs["y"], ft["p"][I_t],
                                                 c_t[I_t].detach(), new_y)

        l_clip, m_clip = L.clip_alignment_loss(fs["q"], fs["y"], ft["q"], cfg["lambda_ent"])
        l_fac = torch.zeros((), device=device)
        if cfg.get("use_factorization", True):
            l_fac_s, _ = L.factorization_loss(fs["z"], fs["Pbar"], fs["C"], fs["V"],
                                              fs["A"], self.factorization.B_s, fs["E"],
                                              cfg.get("beta", 1e-3), cfg.get("xi", 1e-3),
                                              cfg.get("gamma", 1e-2))
            l_fac_t, _ = L.factorization_loss(ft["z"], ft["Pbar"], ft["C"], ft["V"],
                                              ft["A"], self.factorization.B_t, ft["E"],
                                              cfg.get("beta", 1e-3), cfg.get("xi", 1e-3),
                                              cfg.get("gamma", 1e-2))
            l_text = L.text_subspace_loss(fs["V"], self.V_T)
            l_fac = l_fac_s + l_fac_t + cfg.get("alpha", 1.0) * l_text
        l_dom = torch.zeros((), device=device)
        l_leak = torch.zeros((), device=device)
        if cfg.get("use_factorization", True) and cfg.get("use_disentangle", True):
            l_dom = L.domain_loss(self.D_dom(fs["r"]), self.D_dom(ft["r"]))
            l_leak = L.leakage_loss(self.D_cls(fs["r"]), fs["y"])

        loss = (l_clip + cfg["lambda_fac"] * l_fac
                + cfg["lambda_priv"] * (l_dom + cfg.get("lambda_leak", 0.1) * l_leak)
                + cfg["lambda_cert"] * l_cert
                + cfg["lambda_tail"] * l_tail
                + cfg["lambda_sup"] * l_sup)
        metrics = {"l_clip": l_clip.item(), "l_fac": l_fac.item(),
                   "l_dom": l_dom.item(), "l_leak": l_leak.item(),
                   "l_cert": l_cert.item(), "l_tail": l_tail.item(),
                   "l_sup": l_sup.item(), "cert_frac": float(I_t.sum()) / max(float(ft["z"].shape[0]), 1),
                   "donor_valid_s": float(valid_s.float().mean().item()),
                   "donor_valid_t": float(valid_t.float().mean().item())}
        extra = {"I_t": I_t, "yhat_t": yhat_t, "c_t": c_t, "valid_t": valid_t,
                 "u_s": u_s if cfg.get("use_certificate", True) else torch.zeros_like(valid_s.float()),
                 "u_t": u_t if cfg.get("use_certificate", True) else torch.zeros_like(valid_t.float())}
        return loss, metrics, extra

    # ------------------------------------------------------------ ablations
    def _select_targets(self, ft: dict, c_t: Optional[torch.Tensor],
                        yhat_t: torch.Tensor, valid_t: Optional[torch.Tensor],
                        pi_e: float) -> torch.Tensor:
        """Class-wise selection of the target set (certificate/confidence/random/none)."""
        cfg = self.cfg
        device = ft["z"].device
        n = ft["z"].shape[0]
        sel = cfg.get("selection", "certificate")
        if sel == "none":
            return torch.zeros(n, dtype=torch.bool, device=device)
        if sel == "certificate":
            return select_certified_target(c_t, yhat_t, valid_t, pi_e)
        if sel == "confidence":
            score = ft["q"].max(dim=1).values
        else:  # random
            score = torch.rand(n, device=device)
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        for cls in yhat_t.unique():
            idx = (yhat_t == cls).nonzero(as_tuple=True)[0]
            n_k = int(round(pi_e * idx.numel()))
            if n_k <= 0:
                continue
            _, top = score[idx].topk(min(n_k, idx.numel()))
            keep[idx[top]] = True
        return keep

    @torch.no_grad()
    def _corrupt_edges(self, Q: torch.Tensor, active_anchor_ids: torch.Tensor,
                       frac: float, seed: int) -> torch.Tensor:
        """Rewire a fraction of sample–anchor edges to a same-class anchor.

        Preserves the sample's outgoing weight (row degree) and the endpoint
        type; no labels are used. Used by Table edge_corruption.
        """
        n, M = Q.shape
        Q = Q.clone()
        if M < 2 or n == 0:
            return Q
        rng = np.random.default_rng(seed)
        rows = rng.choice(n, size=int(round(frac * n)), replace=False)
        anchor_cls = (active_anchor_ids // self.cfg["anchors_per_class"]).cpu().numpy()
        for row in rows:
            nz = Q[row].nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                continue
            src_col = int(nz[rng.integers(nz.numel())])
            w = float(Q[row, src_col])
            cand = [c for c in range(M)
                    if anchor_cls[c] == anchor_cls[src_col] and c != src_col
                    and float(Q[row, c]) == 0.0]
            if not cand:
                continue
            dst = int(rng.choice(cand))
            Q[row, src_col] = 0.0
            Q[row, dst] = w
        return Q

    # ---------------------------------------------------------------- training
    def fit(self, class_names: Sequence[str], z_s: torch.Tensor, y_s: torch.Tensor,
            z_t: torch.Tensor, z_t_test: Optional[torch.Tensor] = None,
            y_t_test: Optional[torch.Tensor] = None,
            eval_every: int = 0, log_every: int = 10,
            save_path: Optional[str] = None) -> dict:
        """Full training: warm-up then adaptation.

        ``z_t`` is the adaptation target split (full target set for
        transductive; official target-train for inductive). ``z_t_test`` is
        used only when provided (inductive evaluation is external).
        """
        cfg = self.cfg
        device = self.device
        class_names = list(class_names)
        # T = frozen-text prototypes with the CURRENT trainable prompt; V_T is
        # the rank-r right singular subspace of T. T is recomputed every
        # iteration because the prompt context vectors are trainable.
        T0 = self._text_prototypes(class_names)
        U, S, Rh = torch.linalg.svd(T0, full_matrices=False)
        eps = cfg.get("rank_eps", 1e-3)
        r_eff = int((S >= eps * S[0]).sum().item())
        r_eff = min(r_eff, self.r)
        self.V_T = Rh[:r_eff].t().contiguous().detach()   # (d, r_eff), constant target subspace
        # Initialize the (d, r0) semantic basis with the text subspace plus an
        # orthogonal random completion (Vbar keeps its declared shape r0).
        pert = cfg.get("init_perturb", 1e-3)
        extra = torch.randn_like(self.factorization.Vbar)[:, : self.factorization.Vbar.shape[1] - r_eff]
        extra = extra - self.V_T @ (self.V_T.t() @ extra)  # orthogonalize
        extra = F.normalize(extra, dim=-1)
        self.factorization.Vbar.data = torch.cat([self.V_T, pert * extra], dim=1)

        # Warm-up: fit energy bins on a reservoir.
        with torch.no_grad():
            res_norms = torch.cat([z_s[: min(z_s.shape[0], 2000)].norm(dim=1),
                                   z_t[: min(z_t.shape[0], 2000)].norm(dim=1)])
        # Energy bins are fit on residual norms AFTER first decomposition pass;
        # to honor "fixed from a warm-up reservoir", we fit after warm-up below.

        z_s = z_s.to(device)
        y_s = y_s.to(device)
        z_t = z_t.to(device)

        history = {"train_loss": [], "epoch": [], "acc_test": []}
        total_epochs = cfg["warmup_epochs"] + cfg["adapt_epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6)

        for epoch in range(total_epochs):
            self._run_epoch(class_names, T0, z_s, y_s, z_t, epoch,
                            z_t_test=z_t_test, y_t_test=y_t_test,
                            eval_every=eval_every, log_every=log_every,
                            history=history, save_path=save_path)
            self.scheduler.step()
            history["epoch"].append(epoch + 1)
        return history

    def _run_epoch(self, class_names, T0, z_s, y_s, z_t, epoch,
                   z_t_test=None, y_t_test=None, eval_every=0, log_every=10,
                   history=None, save_path=None) -> None:
        cfg = self.cfg
        device = self.device
        iters = cfg["iterations_per_epoch"]
        sampler_s = FeatureSampler(z_s, y_s, cfg["batch_size"], iters, cfg["seed"] + epoch)
        sampler_t = FeatureSampler(z_t, None, cfg["batch_size"], iters, cfg["seed"] + epoch)
        warmup = epoch < cfg["warmup_epochs"]
        if not warmup:
            pi_e = cfg["pi_min"] + (cfg["pi_max"] - cfg["pi_min"]) * \
                (epoch - cfg["warmup_epochs"]) / max(cfg["adapt_epochs"] - 1, 1)
        else:
            pi_e = cfg["pi_min"]

        for it in range(iters):
            z_b, y_b = next(sampler_s)
            z_tb, _ = next(sampler_t)
            self.optimizer.zero_grad(set_to_none=True)

            # Recomputed per iteration (prompt context vectors are trainable).
            T = self._text_prototypes(class_names)

            fs = self._project(z_b, T, y_b, "s")
            fs["y"] = y_b
            ft = self._project(z_tb, T, None, "t")

            if warmup:
                loss, metrics = self._warmup_loss(fs, ft, T)
                extra = {}
            else:
                loss, metrics, extra = self._adaptation_loss(fs, ft, T, pi_e, epoch)

            loss.backward()
            self.optimizer.step()
            self.step += 1

            # Enqueue detached residual snapshots after every parameter step.
            if not warmup:
                self.queues["s"].push(fs["r"], fs["y"], self.energy_bins.assign(fs["r"].norm(dim=1)).to(device),
                                      torch.zeros(fs["z"].shape[0], device=device))
                self.queues["t"].push(ft["r"], ft["q"].argmax(dim=1),
                                      self.energy_bins.assign(ft["r"].norm(dim=1)).to(device),
                                      compute_margin(ft["q"]))

            if history is not None and it % log_every == 0:
                history["train_loss"].append(loss.item())
                LOGGER.info("epoch %d it %d %s loss=%.4f %s",
                            epoch, it, "warmup" if warmup else "adapt", loss.item(),
                            {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in metrics.items()})

        # End of warm-up: fit energy bins from a reservoir and initialize the
        # anchors with class-wise k-means over labeled source features.
        # (Not wrapped in torch.no_grad: the local solver needs an active
        # autograd graph for its internal grad call; all outputs are detached.)
        if epoch == cfg["warmup_epochs"] - 1:
            fs_res = self._project(z_s[: min(z_s.shape[0], 2000)], T,
                                   y_s[: min(z_s.shape[0], 2000)], "s")
            fs_res["y"] = y_s[: min(z_s.shape[0], 2000)]
            ft_res = self._project(z_t[: min(z_t.shape[0], 2000)], T, None, "t")
            norms = torch.cat([fs_res["r"].norm(dim=1), ft_res["r"].norm(dim=1)])
            self.energy_bins.fit(norms.detach().cpu())
            LOGGER.info("energy bins fit with edges=%s", self.energy_bins.edges)
            self.anchors.initialize_from_source(fs_res["h"].detach(), fs_res["y"])
            LOGGER.info("anchors initialized by class-wise k-means on labeled source")

        if eval_every > 0 and (epoch + 1) % eval_every == 0 and z_t_test is not None:
            acc = self.evaluate(class_names, T, z_t_test, y_t_test)
            if history is not None:
                history["acc_test"].append(acc)
            LOGGER.info("epoch %d eval acc=%.4f", epoch, acc)

        if save_path is not None and (epoch + 1) % max(cfg.get("save_every", 1), 1) == 0:
            self.save(save_path, extra={"epoch": epoch})
            if epoch == cfg["warmup_epochs"] - 1:
                # Training-stage diagnostic (Table certificate_diag) compares the
                # warm-up checkpoint with the final one.
                self.save(os.path.join(os.path.dirname(save_path), "model_warmup.pt"),
                          extra={"epoch": epoch})

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, class_names: Sequence[str], T: torch.Tensor,
                 z_test: torch.Tensor, y_test: Optional[torch.Tensor]) -> float:
        """Frozen classification with the projected posterior p_i."""
        T = T if T is not None else self._text_prototypes(list(class_names))
        V = self.factorization.V()
        C = T @ V
        accs = []
        for i in range(0, z_test.shape[0], 256):
            z = z_test[i: i + 256].to(self.device)
            h = z @ V
            p = torch.softmax(self.text_logit_scale * h @ C.t(), dim=-1)
            pred = p.argmax(dim=1)
            if y_test is not None:
                y = y_test[i: i + 256].to(self.device)
                accs.append(float((pred == y).float().mean().item()))
        if y_test is not None:
            return float(np.mean(accs))
        return float("nan")
