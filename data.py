"""Dataset loading and frozen-CLIP feature caching for the four benchmarks.

Dataset layout expected under ``data_root`` (see ``tools/prepare_data.py`` for
download instructions):

    office31/     amazon/, dslr/, webcam/            (31 classes)
    office_home/  Art/, Clipart/, Product/, Real_World/   (65 classes)
    visda/        train/, validation/                (12 classes)
    domainnet/    clipart/, painting/, real/, sketch/ (345 classes)

DomainNet uses the official train/test text lists. The vision tower is frozen,
so image features are extracted once and cached to ``.pt``; training then runs
on the cached features, matching the paper's frozen-encoder protocol.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

BENCHMARK_DOMAINS = {
    "office31": ["amazon", "dslr", "webcam"],
    "office_home": ["Art", "Clipart", "Product", "Real_World"],
    "visda": ["train", "validation"],
    "domainnet": ["clipart", "painting", "real", "sketch"],
}

BENCHMARK_CLASSES = {
    "office31": 31,
    "office_home": 65,
    "visda": 12,
    "domainnet": 345,
}


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def domainnet_split_files(data_root: str, domain: str) -> Tuple[str, str]:
    """Official DomainNet lists: {domain}_train.txt / {domain}_test.txt.

    Lines are "<path> <label>" (label = class index in the official 345-class
    ordering). If the lists are missing, we fall back to the full directory.
    """
    base = os.path.join(data_root, "domainnet")
    train_list = os.path.join(base, f"{domain}_train.txt")
    test_list = os.path.join(base, f"{domain}_test.txt")
    if os.path.exists(train_list) and os.path.exists(test_list):
        return train_list, test_list
    return "", ""


def _load_domainnet_split(list_path: str, base: str):
    paths, labels = [], []
    for ln in _read_lines(list_path):
        parts = ln.split()
        if len(parts) < 2:
            continue
        p, lab = parts[0], int(parts[1])
        full = os.path.join(base, "domainnet", p)
        paths.append(full)
        labels.append(lab)
    return paths, labels


def _load_imagefolder(root: str):
    """Load (paths, labels, class_names) from a torchvision-style folder."""
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        cdir = os.path.join(root, c)
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(cdir, f))
                labels.append(class_to_idx[c])
    return paths, labels, classes


def load_split(cfg: dict, src_domain: str, tgt_domain: str,
               inductive: bool = False, target_split: str = "train"):
    """Return (paths_s, labels_s, paths_t, labels_t, class_names)."""
    benchmark = cfg["benchmark"]
    data_root = cfg["data_root"]
    if benchmark == "domainnet":
        base = os.path.join(data_root, "domainnet")
        src_paths, src_labels, class_names = _load_imagefolder(os.path.join(base, src_domain))
        if inductive:
            tr_list, te_list = domainnet_split_files(data_root, tgt_domain)
            if tr_list and te_list:
                if target_split == "train":
                    t_paths, t_labels = _load_domainnet_split(tr_list, base)
                else:
                    t_paths, t_labels = _load_domainnet_split(te_list, base)
            else:
                t_paths, t_labels, _ = _load_imagefolder(os.path.join(base, tgt_domain))
        else:
            t_paths, t_labels, _ = _load_imagefolder(os.path.join(base, tgt_domain))
        return src_paths, src_labels, t_paths, t_labels, class_names
    if benchmark == "office31":
        base = os.path.join(data_root, "office31")
        s_paths, s_labels, s_cls = _load_imagefolder(os.path.join(base, src_domain))
        t_paths, t_labels, t_cls = _load_imagefolder(os.path.join(base, tgt_domain))
        class_names = sorted(set(s_cls) | set(t_cls))
        return s_paths, s_labels, t_paths, t_labels, class_names
    if benchmark == "office_home":
        base = os.path.join(data_root, "office_home")
        s_paths, s_labels, s_cls = _load_imagefolder(os.path.join(base, src_domain))
        t_paths, t_labels, t_cls = _load_imagefolder(os.path.join(base, tgt_domain))
        class_names = sorted(set(s_cls) | set(t_cls))
        return s_paths, s_labels, t_paths, t_labels, class_names
    if benchmark == "visda":
        base = os.path.join(data_root, "visda")
        s_paths, s_labels, class_names = _load_imagefolder(os.path.join(base, "train"))
        t_paths, t_labels, _ = _load_imagefolder(os.path.join(base, "validation"))
        return s_paths, s_labels, t_paths, t_labels, class_names
    raise ValueError(f"unknown benchmark {benchmark}")


def clip_transform(size: int = 224):
    try:
        from torchvision import transforms
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision is required for image preprocessing") from e
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


@torch.no_grad()
def extract_feature_cache(cfg: dict, backend: torch.nn.Module,
                          paths: Sequence[str], cache_path: str,
                          device: torch.device) -> torch.Tensor:
    """Extract frozen image features and cache to ``cache_path`` (.pt)."""
    if os.path.exists(cache_path):
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert feats.shape[0] == len(paths), f"cache mismatch {cache_path}"
        return feats.to(device)
    tr = clip_transform(cfg.get("image_size", 224))
    feats = []
    bs = 64
    for i in range(0, len(paths), bs):
        chunk = paths[i: i + bs]
        batch = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            batch.append(tr(img))
        imgs = torch.stack(batch).to(device)
        feats.append(backend.image_features(imgs).cpu())
    feats = torch.cat(feats, dim=0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(feats, cache_path)
    return feats.to(device)


class DataModule:
    """Assembles cached features for one transfer task."""

    def __init__(self, cfg: dict, backend: torch.nn.Module, device: torch.device):
        self.cfg = cfg
        self.backend = backend
        self.device = device

    def prepare(self, src_domain: str, tgt_domain: str,
                inductive: bool = False, cache_dir: Optional[str] = None) -> dict:
        cfg = self.cfg
        cache_dir = cache_dir or os.path.join(cfg["data_root"], ".feat_cache")
        # Target-train / target-test splits for the inductive protocol.
        if inductive:
            s_paths, s_labels, tt_paths, tt_labels, class_names = load_split(
                cfg, src_domain, tgt_domain, inductive=True, target_split="train")
            te_paths, te_labels, _, _, _ = load_split(
                cfg, src_domain, tgt_domain, inductive=True, target_split="test")
            z_s = extract_feature_cache(cfg, self.backend, s_paths,
                                        os.path.join(cache_dir, f"{src_domain}_{cfg['seed']}_s.pt"),
                                        self.device)
            z_tt = extract_feature_cache(cfg, self.backend, tt_paths,
                                         os.path.join(cache_dir, f"{tgt_domain}_train_{cfg['seed']}.pt"),
                                         self.device)
            z_te = extract_feature_cache(cfg, self.backend, te_paths,
                                         os.path.join(cache_dir, f"{tgt_domain}_test_{cfg['seed']}.pt"),
                                         self.device)
            return {"z_s": z_s, "y_s": torch.tensor(s_labels, device=self.device),
                    "z_t": z_tt, "y_t": None,
                    "z_t_test": z_te, "y_t_test": torch.tensor(te_labels, device=self.device),
                    "class_names": class_names}
        s_paths, s_labels, t_paths, t_labels, class_names = load_split(
            cfg, src_domain, tgt_domain, inductive=False)
        z_s = extract_feature_cache(cfg, self.backend, s_paths,
                                    os.path.join(cache_dir, f"{src_domain}_{cfg['seed']}_s.pt"),
                                    self.device)
        z_t = extract_feature_cache(cfg, self.backend, t_paths,
                                    os.path.join(cache_dir, f"{tgt_domain}_{cfg['seed']}_t.pt"),
                                    self.device)
        return {"z_s": z_s, "y_s": torch.tensor(s_labels, device=self.device),
                "z_t": z_t, "y_t": torch.tensor(t_labels, device=self.device),
                "z_t_test": None, "y_t_test": None,
                "class_names": class_names}
