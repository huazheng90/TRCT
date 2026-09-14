"""Post-hoc diagnostics: ranking, coverage-risk, graph quality, probes,
additivity, transport plausibility, and the Appendix C construction checks.

All diagnostics load a frozen checkpoint (or two, for the training-stage
comparison) and operate on cached features. Target labels are opened only
post hoc. No diagnostic retrains the model.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.certificate import EnergyBins, compute_margin
from src.graph import build_certified_graph, graph_purity_and_smoothing, spectral_tail_loss
from src.model import AnchorBank, TextAnchoredFactorization
from src.trainer import TCRTTrainer
from src.utils import LOGGER, auroc_fast, point_biserial, set_seed


@torch.no_grad()
def decompose_all(trainer: TCRTTrainer, z: torch.Tensor, T: torch.Tensor,
                  y: Optional[torch.Tensor], domain: str, batch: int = 256) -> Dict[str, torch.Tensor]:
    """Decompose a full feature matrix; returns concatenated tensors."""
    outs = {"z": [], "h": [], "p": [], "q": [], "s": [], "r": [], "A": [], "E": []}
    for i in range(0, z.shape[0], batch):
        zb = z[i: i + batch]
        yb = y[i: i + batch] if y is not None else None
        f = trainer._project(zb, T, yb, domain)
        for k in outs:
            outs[k].append(f[k].cpu())
    return {k: torch.cat(v, dim=0) for k, v in outs.items()}


def _donor_pool(r: torch.Tensor, cls: torch.Tensor, bins: torch.Tensor,
                margin: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"r": r, "cls": cls, "bin": bins, "margin": margin}


def score_queries(trainer, z_q, T, y_q, donor, donor_is_target,
                  cfg, cvarr: bool = True) -> Dict[str, torch.Tensor]:
    """Compute certificate/margin/maxsoftmax/u for a query set.

    ``donor`` is a dict from _donor_pool for the opposite domain.
    """
    from src.certificate import compute_certificate_batch
    fq = decompose_all(trainer, z_q, T, None, "t")
    yhat = fq["q"].argmax(dim=1)
    bins = trainer.energy_bins.assign(fq["r"].norm(dim=1)).long()
    u, c, valid, cnt = compute_certificate_batch(
        fq["z"], fq["s"], fq["r"], yhat, T, trainer.text_logit_scale,
        donor["r"], donor["cls"], donor["bin"], bins, donor["margin"],
        donor_is_target=donor_is_target,
        kappa_donor=cfg.get("kappa_donor", 0.05),
        kappa_r=cfg.get("kappa_r", 2.0),
        j_min=cfg.get("j_min", 1),
        rho=cfg.get("rho", 0.1), eta=cfg.get("eta", 0.1),
        kappa=cfg.get("kappa_cert", 0.0), zeta=cfg.get("zeta", 0.1),
    )
    m = compute_margin(fq["q"])
    maxsp = fq["q"].max(dim=1).values
    return {"cert": c.detach(), "margin": m.detach(), "maxsoftmax": maxsp.detach(),
            "u": u.detach(), "valid": valid, "yhat": yhat.detach(), "counts": cnt}


def _classwise_precision(score: torch.Tensor, correct: torch.Tensor,
                         yhat: torch.Tensor, coverage: float) -> float:
    n = score.shape[0]
    sel = torch.zeros(n, dtype=torch.bool, device=score.device)
    for c in torch.unique(yhat).tolist():
        idx = (yhat == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = max(1, int(np.ceil(coverage * idx.numel())))
        k = min(k, idx.numel())
        top = torch.topk(score[idx], k).indices
        sel[idx[top]] = True
    if sel.sum() == 0:
        return 0.0
    return float(correct[sel].float().mean().item())


def ranking_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Table ranking: point-biserial, AUROC, AURC, precision at 80% coverage.

    All scores are computed on the same frozen checkpoint. The donor bank is
    the labeled source reservoir reconstructed with the checkpoint rules.
    """
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])

    scores = {k: [] for k in ["cert", "margin", "maxsoftmax", "u"]}
    aurocs, aurcs, precs, pbs = {}, {}, {}, {}
    for seed in cfg["seeds"]:
        set_seed(seed)
        ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
        trainer = TCRTTrainer(cfg, backend, device)
        trainer.load(ckpt, load_optimizer=False)
        # Rebuild energy bins from the checkpoint rules.
        with torch.no_grad():
            fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
            ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
        eb = EnergyBins(cfg.get("energy_bins", 10))
        eb.fit(torch.cat([fs_res["r"].norm(dim=1), ft_res["r"].norm(dim=1)]).cpu())
        trainer.energy_bins = eb
        donor = _donor_pool(fs_res["r"], data["y_s"],
                            eb.assign(fs_res["r"].norm(dim=1)).long(),
                            torch.zeros(fs_res["r"].shape[0]))
        sq = score_queries(trainer, data["z_t"], T, None, donor,
                           donor_is_target=False, cfg=cfg, cvarr=True)
        correct = (sq["yhat"] == data["y_t"]).float()
        # No-tail transport stability: mean KL per query (u without CVaR).
        u_mean = _mean_transport_kl(trainer, data["z_t"], T, donor, eb, cfg)
        for k in scores:
            scores[k].append(sq[k])
        scores["u"].append(u_mean)

        for name, s in [("certificate", sq["cert"]), ("margin", sq["margin"]),
                        ("maxsoftmax", sq["maxsoftmax"]), ("transport_mean", u_mean),
                        ("u_cvar", sq["u"])]:
            s = s.detach()
            sc = s[valid := sq["valid"]].numpy() if name != "transport_mean" else s.numpy()
            corr = correct[sq["valid"]].numpy() if name != "transport_mean" else correct.numpy()
            aurocs.setdefault(name, []).append(auroc_fast(sc, corr))
            precs.setdefault(name, []).append(
                _classwise_precision(s, correct, sq["yhat"], 0.8) * 100)
    out = {"seeds": cfg["seeds"]}
    for name in ["certificate", "margin", "maxsoftmax", "transport_mean", "u_cvar"]:
        out[name] = {"auroc_mean": float(np.mean(aurocs[name])),
                     "auroc_std": float(np.std(aurocs[name], ddof=1)),
                     "precision80_mean": float(np.mean(precs[name])),
                     "precision80_std": float(np.std(precs[name], ddof=1))}
    LOGGER.info("ranking diagnostic: %s", out)
    return out


def _mean_transport_kl(trainer, z_q, T, donor, eb, cfg):
    """Mean (not CVaR) KL over donors, used by the "no tail aggregation" row."""
    fq = decompose_all(trainer, z_q, T, None, "t")
    yhat = fq["q"].argmax(dim=1)
    bins = eb.assign(fq["r"].norm(dim=1)).long()
    P_q = torch.softmax(trainer.text_logit_scale * fq["z"] @ T.t(), dim=-1)
    u = torch.zeros(z_q.shape[0])
    for i in range(z_q.shape[0]):
        same = (donor["cls"] == yhat[i])
        same_bin = (donor["bin"] == bins[i])
        s_n = fq["s"][i].norm().clamp(min=1e-8)
        ratio = donor["r"].norm(dim=1) / s_n
        idx = (same & same_bin & (ratio <= cfg.get("kappa_r", 2.0))).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        z_t = F.normalize(fq["s"][i].unsqueeze(0) + donor["r"][idx], dim=-1)
        P_t = torch.softmax(trainer.text_logit_scale * z_t @ T.t(), dim=-1)
        q = P_q[i]
        u[i] = (q * torch.log(q.clamp(min=1e-12) / P_t.clamp(min=1e-12))).sum(dim=-1).mean()
    return u


def coverage_risk_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Fig. coverage_risk: risk = 1 - precision over a coverage sweep."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    grid = np.arange(0.1, 1.0 + 1e-9, 0.1)
    curves = {k: [] for k in ["cert", "margin", "maxsoftmax"]}
    for seed in cfg["seeds"]:
        set_seed(seed)
        ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
        trainer = TCRTTrainer(cfg, backend, device)
        trainer.load(ckpt, load_optimizer=False)
        with torch.no_grad():
            fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
        eb = EnergyBins(cfg.get("energy_bins", 10))
        eb.fit(fs_res["r"].norm(dim=1).cpu())
        trainer.energy_bins = eb
        donor = _donor_pool(fs_res["r"], data["y_s"], eb.assign(fs_res["r"].norm(dim=1)).long(),
                            torch.zeros(fs_res["r"].shape[0]))
        sq = score_queries(trainer, data["z_t"], T, None, donor,
                           donor_is_target=False, cfg=cfg)
        correct = (sq["yhat"] == data["y_t"]).float()
        for name in curves:
            risks = []
            for pi in grid:
                prec = _classwise_precision(sq[name], correct, sq["yhat"], float(pi))
                risks.append(1.0 - prec)
            curves[name].append(risks)
    out = {"coverage": [float(p) for p in grid]}
    for name, runs in curves.items():
        arr = np.asarray(runs)  # (seeds, len(grid))
        out[name] = {"mean": [float(v) for v in arr.mean(axis=0)],
                     "std": [float(v) for v in arr.std(axis=0, ddof=1)]}
    return out


def stage_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Table certificate_diag: warm-up vs final checkpoint on the common cohort."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    res = {"seeds": cfg["seeds"]}
    for stage in ["warmup", "final"]:
        auroc_c, auroc_m, prec_c, prec_m, u_corr, u_inc = [], [], [], [], [], []
        frac_valid = []
        for seed in cfg["seeds"]:
            set_seed(seed)
            ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}",
                                "model_warmup.pt" if stage == "warmup" else "model.pt")
            trainer = TCRTTrainer(cfg, backend, device)
            trainer.load(ckpt, load_optimizer=False)
            with torch.no_grad():
                fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
            eb = EnergyBins(cfg.get("energy_bins", 10))
            eb.fit(fs_res["r"].norm(dim=1).cpu())
            trainer.energy_bins = eb
            donor = _donor_pool(fs_res["r"], data["y_s"],
                                eb.assign(fs_res["r"].norm(dim=1)).long(),
                                torch.zeros(fs_res["r"].shape[0]))
            sq = score_queries(trainer, data["z_t"], T, None, donor,
                               donor_is_target=False, cfg=cfg)
            v = sq["valid"]
            frac_valid.append(float(v.mean().item()))
            correct = (sq["yhat"] == data["y_t"]).float()
            auroc_c.append(auroc_fast(sq["cert"][v].numpy(), correct[v].numpy()))
            auroc_m.append(auroc_fast(sq["margin"][v].numpy(), correct[v].numpy()))
            prec_c.append(_classwise_precision(sq["cert"], correct, sq["yhat"], 0.8) * 100)
            prec_m.append(_classwise_precision(sq["margin"], correct, sq["yhat"], 0.8) * 100)
            u_corr.append(float(sq["u"][v][correct[v] == 1].mean().item()))
            u_inc.append(float(sq["u"][v][correct[v] == 0].mean().item()))
        res[stage] = {
            "donor_valid_frac": (float(np.mean(frac_valid)), float(np.std(frac_valid, ddof=1))),
            "cert_auroc": (float(np.mean(auroc_c)), float(np.std(auroc_c, ddof=1))),
            "margin_auroc": (float(np.mean(auroc_m)), float(np.std(auroc_m, ddof=1))),
            "cert_precision80": (float(np.mean(prec_c)), float(np.std(prec_c, ddof=1))),
            "margin_precision80": (float(np.mean(prec_m)), float(np.std(prec_m, ddof=1))),
            "u_correct": (float(np.mean(u_corr)), float(np.std(u_corr, ddof=1))),
            "u_incorrect": (float(np.mean(u_inc)), float(np.std(u_inc, ddof=1))),
        }
    return res


def graph_quality_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Table graph_quality: purity / effective degree / smoothed accuracy."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    out = {"seeds": cfg["seeds"]}
    for subset in ["all", "random", "confidence", "certificate"]:
        pur, deg, acc = [], [], []
        for seed in cfg["seeds"]:
            set_seed(seed)
            ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
            trainer = TCRTTrainer(cfg, backend, device)
            trainer.load(ckpt, load_optimizer=False)
            with torch.no_grad():
                fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
                ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
            eb = EnergyBins(cfg.get("energy_bins", 10))
            eb.fit(fs_res["r"].norm(dim=1).cpu())
            trainer.energy_bins = eb
            donor = _donor_pool(fs_res["r"], data["y_s"],
                                eb.assign(fs_res["r"].norm(dim=1)).long(),
                                torch.zeros(fs_res["r"].shape[0]))
            sq = score_queries(trainer, data["z_t"], T, None, donor,
                               donor_is_target=False, cfg=cfg)
            yhat = sq["yhat"]
            n = yhat.shape[0]
            if subset == "all":
                sel = torch.ones(n, dtype=torch.bool)
            elif subset == "random":
                rng = np.random.default_rng(seed)
                sel = torch.zeros(n, dtype=torch.bool)
                for c in torch.unique(yhat).tolist():
                    idx = (yhat == c).nonzero(as_tuple=True)[0]
                    k = max(1, int(np.ceil(0.8 * idx.numel())))
                    pick = rng.choice(idx.numel(), size=min(k, idx.numel()), replace=False)
                    sel[idx[torch.tensor(pick, dtype=torch.long)]] = True
            elif subset == "confidence":
                sel = _classwise_selector(sq["maxsoftmax"], yhat, 0.8)
            else:
                sel = _classwise_selector(sq["cert"], yhat, 0.8)
            if sel.sum() == 0:
                continue
            h_t = ft_res["h"][sel]
            y_t = yhat[sel]
            labels = data["y_t"][sel]
            p, e, a = graph_purity_and_smoothing(h_t, y_t, labels, trainer.anchors,
                                                 cfg.get("tau_g", 10.0))
            pur.append(p * 100), deg.append(e), acc.append(a * 100)
        out[subset] = {"purity": (float(np.mean(pur)), float(np.std(pur, ddof=1))),
                       "effective_degree": (float(np.mean(deg)), float(np.std(deg, ddof=1))),
                       "smoothed_acc": (float(np.mean(acc)), float(np.std(acc, ddof=1)))}
    return out


def _classwise_selector(score: torch.Tensor, yhat: torch.Tensor, cov: float) -> torch.Tensor:
    n = score.shape[0]
    sel = torch.zeros(n, dtype=torch.bool, device=score.device)
    for c in torch.unique(yhat).tolist():
        idx = (yhat == c).nonzero(as_tuple=True)[0]
        k = max(1, int(np.ceil(cov * idx.numel())))
        k = min(k, idx.numel())
        top = torch.topk(score[idx], k).indices
        sel[idx[top]] = True
    return sel


def factorization_probe_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Table factorization_probe: linear probes + normalized component energy."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    out = {"seeds": cfg["seeds"]}
    reps = ["z", "s", "r", "E"]
    for rep in reps:
        cls_acc, dom_acc, energy = [], [], []
        for seed in cfg["seeds"]:
            set_seed(seed)
            ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
            trainer = TCRTTrainer(cfg, backend, device)
            trainer.load(ckpt, load_optimizer=False)
            with torch.no_grad():
                fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
                ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
            # Class probe: trained on labeled source, tested on disjoint source split.
            idx = torch.randperm(fs_res["z"].shape[0], generator=torch.Generator().manual_seed(seed))
            n_tr = int(0.8 * fs_res["z"].shape[0])
            tr, te = idx[:n_tr], idx[n_tr:]
            Xtr, ytr = fs_res[rep][tr], data["y_s"][tr]
            Xte, yte = fs_res[rep][te], data["y_s"][te]
            clf = _linear_probe(Xtr, ytr, Xte, yte, device)
            cls_acc.append(clf)
            # Domain probe: balanced source/target, domain identity.
            n_d = min(500, fs_res[rep].shape[0], ft_res[rep].shape[0])
            Xd = torch.cat([fs_res[rep][:n_d], ft_res[rep][:n_d]])
            yd = torch.cat([torch.zeros(n_d, dtype=torch.long), torch.ones(n_d, dtype=torch.long)])
            perm = torch.randperm(Xd.shape[0], generator=torch.Generator().manual_seed(seed + 1))
            n_tr2 = int(0.8 * Xd.shape[0])
            dom_acc.append(_linear_probe(Xd[perm[:n_tr2]], yd[perm[:n_tr2]],
                                         Xd[perm[n_tr2:]], yd[perm[n_tr2:]], device))
            energy.append(float((fs_res[rep].norm(dim=1) ** 2).mean().item()))
        base = float((fs_res["z"].norm(dim=1) ** 2).mean().item())
        out[rep] = {"class_acc": (float(np.mean(cls_acc)), float(np.std(cls_acc, ddof=1))),
                    "domain_acc": (float(np.mean(dom_acc)), float(np.std(dom_acc, ddof=1))),
                    "energy_ratio": (float(np.mean(energy)) / base if base > 0 else 0.0)}
    return out


def _linear_probe(Xtr, ytr, Xte, yte, device) -> float:
    from sklearn.linear_model import LogisticRegression
    Xtr = F.normalize(Xtr.float(), dim=-1).cpu().numpy()
    Xte = F.normalize(Xte.float(), dim=-1).cpu().numpy()
    clf = LogisticRegression(max_iter=2000, C=1e-2)
    clf.fit(Xtr, ytr.cpu().numpy())
    return float(clf.score(Xte, yte.cpu().numpy()))


def additivity_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Table additivity_reconstruction: e_full / e_core mean, median, P90."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    all_full, all_core = [], []
    for seed in cfg["seeds"]:
        set_seed(seed)
        ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
        trainer = TCRTTrainer(cfg, backend, device)
        trainer.load(ckpt, load_optimizer=False)
        with torch.no_grad():
            ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
        z = ft_res["z"]
        e_full = (z - (ft_res["s"] + ft_res["r"] + ft_res["E"])).norm(dim=1) / z.norm(dim=1)
        e_core = (z - (ft_res["s"] + ft_res["r"])).norm(dim=1) / z.norm(dim=1)
        all_full.append(e_full.numpy())
        all_core.append(e_core.numpy())
    full = np.concatenate(all_full)
    core = np.concatenate(all_core)
    return {"e_full": {"mean": float(full.mean()), "median": float(np.median(full)),
                       "p90": float(np.percentile(full, 90))},
            "e_core": {"mean": float(core.mean()), "median": float(np.median(core)),
                       "p90": float(np.percentile(core, 90))}}


def transport_plausibility_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Tables transport_plausibility / transport_semantics: six interventions."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    interventions = ["self", "same_src", "same_tgt", "cross_tgt",
                     "random_iso", "random_sub"]
    out = {iv: {"nn_ratio": [], "nn_agree": [], "retention": [],
                "probe": [], "median_kl": []} for iv in interventions}
    for seed in cfg["seeds"]:
        set_seed(seed)
        ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
        trainer = TCRTTrainer(cfg, backend, device)
        trainer.load(ckpt, load_optimizer=False)
        with torch.no_grad():
            fs_res = decompose_all(trainer, data["z_s"], T, data["y_s"], "s")
            ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
        eb = EnergyBins(cfg.get("energy_bins", 10))
        eb.fit(fs_res["r"].norm(dim=1).cpu())
        trainer.energy_bins = eb
        # Source queries, target donors (same queue/pseudo-label rule).
        n_q = min(1000, fs_res["z"].shape[0])
        q_idx = torch.randperm(fs_res["z"].shape[0], generator=torch.Generator().manual_seed(seed))[:n_q]
        yhat_t = ft_res["q"].argmax(dim=1)
        t_bins = eb.assign(ft_res["r"].norm(dim=1)).long()
        # Reference bank: disjoint real features (target).
        ref_bank = ft_res["z"]
        # Source probe: train once on untransported source features.
        probe = _probe_acc(fs_res["z"], data["y_s"], device)
        # Residual-subspace basis from target residuals (for the subspace control).
        R_sub = ft_res["r"][: min(ft_res["r"].shape[0], 2000)]
        _, _, Vt = torch.linalg.svd(R_sub, full_matrices=False)  # (n, d) -> (d, d)
        k_sub = min(16, Vt.shape[1])

        for i in q_idx.tolist():
            z0 = fs_res["z"][i]
            s_i = fs_res["s"][i]
            r_i = fs_res["r"][i]
            cls = int(data["y_s"][i])
            same_cls_t = (yhat_t == cls)
            same_bin_t = (t_bins == eb.assign(r_i.norm().unsqueeze(0)).long()[0])
            s_n = s_i.norm().clamp(min=1e-8)
            ratio_t = ft_res["r"].norm(dim=1) / s_n
            tgt_donor = same_cls_t & same_bin_t & (ratio_t <= cfg.get("kappa_r", 2.0))
            tgt_idx = tgt_donor.nonzero(as_tuple=True)[0]
            cross_idx = (~same_cls_t).nonzero(as_tuple=True)[0]
            if tgt_idx.numel() == 0:
                continue
            j = int(tgt_idx[torch.randint(tgt_idx.numel(), (1,))[0]])
            r_j = ft_res["r"][j]
            src_same = (data["y_s"] == cls).nonzero(as_tuple=True)[0]
            src_same = src_same[src_same != i]
            rng = np.random.default_rng(seed + i)
            for iv in interventions:
                if iv == "self":
                    r_use = r_i
                elif iv == "same_src":
                    if src_same.numel() == 0:
                        continue
                    r_use = fs_res["r"][int(src_same[rng.integers(0, src_same.numel())])]
                elif iv == "same_tgt":
                    r_use = r_j
                elif iv == "cross_tgt":
                    if cross_idx.numel() == 0:
                        continue
                    r_use = ft_res["r"][int(cross_idx[rng.integers(0, cross_idx.numel())])]
                elif iv == "random_iso":
                    r_use = F.normalize(torch.randn_like(r_i), dim=-1) * r_i.norm()
                else:  # random_sub: random direction in the target residual subspace
                    coeff = torch.randn(k_sub, device=device)
                    coeff = coeff / coeff.norm()
                    r_use = (Vt[:k_sub].t() @ coeff) * r_i.norm()
                z_t = F.normalize(s_i + r_use, dim=-1)
                d0 = (1.0 - z0 @ ref_bank.t()).topk(5).values.mean()
                d1 = (1.0 - z_t @ ref_bank.t()).topk(5).values.mean()
                out[iv]["nn_ratio"].append(float(d1 / max(float(d0), 1e-8)))
                nn_cls = int(torch.argmax(z_t @ ref_bank.t()).item())
                out[iv]["nn_agree"].append(float(int(data["y_t"][nn_cls]) == cls))
                P0 = torch.softmax(trainer.text_logit_scale * z0 @ T.t(), dim=-1)
                P1 = torch.softmax(trainer.text_logit_scale * z_t @ T.t(), dim=-1)
                out[iv]["retention"].append(float(P0.argmax() == P1.argmax()))
                out[iv]["median_kl"].append(
                    float((P0 * torch.log(P0.clamp(min=1e-12) / P1.clamp(min=1e-12))).sum().item()))
                out[iv]["probe"].append(float(probe(F.normalize(z_t, dim=-1).unsqueeze(0),
                                                    torch.tensor([cls]))))
    summary = {}
    for iv in interventions:
        summary[iv] = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                       for k, v in out[iv].items() if v}
    return summary


def _probe_acc(X, y, device):
    from sklearn.linear_model import LogisticRegression
    Xn = F.normalize(X.float(), dim=-1).cpu().numpy()
    clf = LogisticRegression(max_iter=2000, C=1e-2)
    clf.fit(Xn, y.cpu().numpy())
    def predict(z, yb):
        p = clf.predict(z.float().cpu().numpy())
        return float((p == yb.cpu().numpy()).mean())
    return predict


def property_checks(cfg: dict, device: torch.device, n_pairs: int = 2000) -> Dict:
    """Appendix C implementation checks: subspace identity, margin condition,
    spectral inequality."""
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=False)
    T = backend.text_prototypes(data["class_names"])
    out = {}
    for seed in cfg["seeds"][:1]:
        ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
        trainer = TCRTTrainer(cfg, backend, device)
        trainer.load(ckpt, load_optimizer=False)
        V = trainer.factorization.V()
        # Subspace identity.
        VT = trainer.V_T
        dev = float((V @ V.t() - VT @ VT.t()).pow(2).sum().item())
        l_text = float((V @ V.t() - VT @ VT.t()).pow(2).sum().item())
        out["subspace"] = {"||VV^T - VT VT^T||_F^2": dev,
                           "orthonormality_err": float((V.t() @ V - torch.eye(V.shape[1], device=device)).abs().max().item())}
        # Margin condition on random donor-valid pairs.
        with torch.no_grad():
            ft_res = decompose_all(trainer, data["z_t"], T, None, "t")
        L_T = trainer.text_logit_scale * torch.linalg.norm(T, 2).item()
        eligible = 0
        satisfied = 0
        rng = np.random.default_rng(seed)
        for _ in range(n_pairs):
            i = int(rng.integers(0, ft_res["z"].shape[0]))
            j = int(rng.integers(0, ft_res["z"].shape[0]))
            if i == j:
                continue
            z0 = ft_res["z"][i]
            z1 = F.normalize(ft_res["s"][i] + ft_res["r"][j], dim=-1)
            l0 = trainer.text_logit_scale * z0 @ T.t()
            l1 = trainer.text_logit_scale * z1 @ T.t()
            gap = float(torch.topk(l0, 2).values.diff().abs().item())
            dist = float((z0 - z1).norm().item())
            eligible += 1
            if dist < gap / (2 * L_T):
                satisfied += 1
                if torch.argmax(l0) != torch.argmax(l1):
                    out["margin_violations"] = out.get("margin_violations", 0) + 1
        out["margin"] = {"eligible_pairs": eligible, "fraction_satisfying": float(satisfied / max(eligible, 1)),
                         "L_T": float(L_T)}
        # Spectral consistency on a small certified graph: the sparse
        # multiplication used by L_tail must equal the dense L_c = I - W
        # construction, and L_c's spectrum must lie in the Laplacian range.
        Q, inv_sqrt_dx, inv_da, _ = build_certified_graph(
            ft_res["h"][:512], ft_res["q"].argmax(dim=1)[:512], trainer.anchors,
            cfg.get("tau_g", 10.0))
        if Q is not None:
            n = Q.shape[0]
            eps = cfg.get("eps_deg", 1e-6)
            W = (inv_sqrt_dx.unsqueeze(1) * Q) @ (inv_da.unsqueeze(1) * Q.t())
            W = W * inv_sqrt_dx.unsqueeze(0)
            Lc = torch.eye(n, device=device) - 0.5 * (W + W.t())
            evals = torch.linalg.eigvalsh(Lc)
            P = ft_res["p"][:n]
            l_tail_sparse = spectral_tail_loss(P, Q, inv_sqrt_dx, inv_da, p=1)
            l_tail_dense = float((P * (Lc @ P)).sum().item()) / max(n, 1)
            out["spectral"] = {
                "lambda_min": float(evals.min()),
                "lambda_max": float(evals.max()),
                "trace_Lc_over_n": float(evals.sum().item()) / n,
                "sparse_dense_max_abs_diff": float(abs(l_tail_sparse - l_tail_dense)),
            }
    return out
