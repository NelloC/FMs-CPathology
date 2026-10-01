"""Export the split manifests: one CSV per dataset and seed with the partition of every indexed image.

Columns: path (relative to the data root), label, group, partition (train / val / test, or 'excluded' for
tiles removed by the tissue filter, splits.TISSUE_MIN). Official partitions (BRACS, SICAPv2) are the same for
every seed and are written once, as seed 'official'. A SHA256SUMS file covers every manifest.

Output: manifests/<dataset>/seed<k>.csv.gz (gzip), manifests/SHA256SUMS
Usage: python pipeline/export_manifests.py [dataset ...]
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CODE_DIR  # noqa: E402
from data_index import load_index  # noqa: E402
from probe import SEEDS  # noqa: E402
from splits import OFFICIAL, check_disjoint, split_rows, usable  # noqa: E402

DATASETS = ["breakhis", "lung", "bach", "hubmap", "sicap", "nct", "panda", "bracs"]
OUT = os.path.join(CODE_DIR, "manifests")


def manifest(dataset, seed):
    idx = load_index(dataset)
    keep = usable(dataset, idx)
    df = idx[keep]
    parts = split_rows(df, dataset, seed)
    check_disjoint(df, parts)
    part = np.full(len(idx), "excluded", dtype=object)
    rows = np.flatnonzero(keep)
    for name, r in parts.items():
        part[rows[r]] = name
    return idx[["path", "label", "group"]].assign(partition=part)


def main():
    for dataset in sys.argv[1:] or DATASETS:
        seeds = ["official"] if dataset in OFFICIAL else SEEDS
        os.makedirs(os.path.join(OUT, dataset), exist_ok=True)
        for seed in seeds:
            m = manifest(dataset, 42 if seed == "official" else seed)
            path = os.path.join(OUT, dataset, f"seed{seed}.csv.gz")
            m.to_csv(path + ".tmp", index=False, compression={"method": "gzip", "mtime": 0})
            os.replace(path + ".tmp", path)
            print(f"{dataset:9s} seed {seed}: " + ", ".join(f"{k} {v}" for k, v in m.partition.value_counts().items()),
                  flush=True)
    sums = []
    for root, _, files in sorted(os.walk(OUT)):
        for f in sorted(files):
            if f.endswith(".csv.gz"):
                p = os.path.join(root, f)
                sums.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}")
    open(os.path.join(OUT, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
