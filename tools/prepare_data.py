#!/usr/bin/env python3
"""Prepare the four benchmark datasets under ``./data``.

The CLIP ViT-B/16 weights and the benchmark images are external assets that are
NOT bundled with this repository (same policy as the reference release). This
script:

1. prints the expected directory layout,
2. downloads the official DomainNet train/test lists and the 345-class label
   file when ``--domainnet-lists`` is given,
3. validates an existing ``data`` tree and reports per-domain image counts.

DomainNet list URLs (official 0.6M release):
    http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart_train.txt
    ... (see https://ai.bu.edu/M3SDA/)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

LAYOUT = """Expected layout under DATA_ROOT (default ./data):

office31/
    amazon/   class_name/*.jpg
    dslr/     class_name/*.jpg
    webcam/   class_name/*.jpg

office_home/
    Art/          class_name/*.jpg
    Clipart/      class_name/*.jpg
    Product/      class_name/*.jpg
    Real_World/   class_name/*.jpg

visda/
    train/        class_name/*.jpg      (synthetic, labeled source)
    validation/   class_name/*.jpg      (real, unlabeled target)

domainnet/
    clipart/   <file>.<ext>             (per-class subfolders optional; the
    painting/   official train/test lists are required for the inductive split)
    real/
    sketch/
    clipart_train.txt   clipart_test.txt
    painting_train.txt  painting_test.txt
    real_train.txt      real_test.txt
    sketch_train.txt    sketch_test.txt
    labels.txt                          (345 class names, one per line)

Download sources:
    Office-31  https://www.cc.gatech.edu/~saghar/?page_id=32
    Office-Home  https://www.hemanthdv.org/officeHomeDataset.html
    VisDA-2017  https://github.com/VisionLearningGroup/taskcv-2017-public
    DomainNet   https://ai.bu.edu/M3SDA/
"""

DOMAINS = ["clipart", "painting", "real", "sketch"]
LIST_BASE = "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/"


def download_domainnet_lists(data_root: str) -> None:
    dn = os.path.join(data_root, "domainnet")
    os.makedirs(dn, exist_ok=True)
    for d in DOMAINS:
        for split in ("train", "test"):
            url = f"{LIST_BASE}{d}_{split}.txt"
            out = os.path.join(dn, f"{d}_{split}.txt")
            if os.path.exists(out):
                print(f"exists: {out}")
                continue
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, out)
    # 345-class label file.
    out = os.path.join(dn, "labels.txt")
    if not os.path.exists(out):
        url = f"{LIST_BASE}labels.txt"
        print(f"downloading {url}")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:  # some mirrors call it label.txt
            print(f"labels.txt failed ({e}); class names fall back to dirs")


def validate(data_root: str) -> None:
    import PIL.Image
    counts = {}
    for bench in ["office31", "office_home", "visda", "domainnet"]:
        base = os.path.join(data_root, bench)
        if not os.path.isdir(base):
            print(f"[missing] {base}")
            continue
        n = 0
        for root, _, files in os.walk(base):
            n += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        counts[bench] = n
        print(f"[{bench}] {n} images")
    if counts.get("office_home", 0) < 10000:
        print("\nwarning: Office-Home looks incomplete (expected 15,588)")
    if counts.get("visda", 0) < 50000:
        print("\nwarning: VisDA-2017 looks incomplete (expected ~208k)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--domainnet-lists", action="store_true",
                   help="download official DomainNet train/test lists")
    p.add_argument("--validate", action="store_true", help="validate existing tree")
    args = p.parse_args()
    if args.domainnet_lists:
        download_domainnet_lists(args.data_root)
    print(LAYOUT)
    if args.validate or args.domainnet_lists:
        validate(args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
