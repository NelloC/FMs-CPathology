"""Train/validation/test row indices per dataset and seed, from the index (pipeline/data_index.py).

Official partitions (BRACS, SICAPv2): train / val / test as distributed; identical for every seed.
Other datasets: GroupShuffleSplit (test_size 0.2, random_state = seed) on the path-sorted list; stratified
split when no group exists. PANDA, BACH and HuBMAP first drop non-tissue tiles (TISSUE_MIN). The 'val' part, used only to choose the probe's
regularization, is an inner split of the outer training part made the same way.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from common import INDEX_DIR

OFFICIAL = {"bracs", "sicap"}

# Tiles we cut ourselves from larger images must contain >= 10% tissue (pipeline/tissue.py): PANDA's tiling
# produced 187k black/white padding tiles (38% of tiles below 10% tissue), BACH and HuBMAP a few blank
# background tiles. Curated datasets are used as distributed; NCT-CRC-HE-100K even has a background class.
TISSUE_MIN = {"panda": 0.10, "bach": 0.10, "hubmap": 0.10}


def usable(dataset, df):
    """Boolean mask over the index rows that enter the analysis."""
    if dataset not in TISSUE_MIN:
        return np.ones(len(df), bool)
    t = pd.read_csv(os.path.join(INDEX_DIR, f"tissue_{dataset}.csv"))
    if t.path.tolist() != df.path.tolist():
        raise RuntimeError(f"tissue_{dataset}.csv does not match the index: run pipeline/tissue.py {dataset}")
    return t.tissue_frac.values >= TISSUE_MIN[dataset]


def filter_config(dataset):
    return {"tissue_min": TISSUE_MIN.get(dataset)}


def data_signature(dataset, df):
    """Hash of everything that defines the analysis input of a dataset: the usable rows' paths, labels,
    groups and partitions, and the filter. A stored result with another signature is stale."""
    import hashlib
    h = hashlib.sha256(repr(filter_config(dataset)).encode())
    for col in ("path", "label", "group", "partition"):
        h.update("\n".join(map(str, df[col].values)).encode())
    return h.hexdigest()[:16]


def _split(rows, labels, groups, seed):
    """(train_rows, held_out_rows): 20% of groups held out, or a stratified 20% when there are no groups."""
    if groups is not None and len(set(groups)) > 1:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(rows, labels, groups=groups))
    else:
        tr, te = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=seed, stratify=labels)
    return rows[np.sort(tr)], rows[np.sort(te)]


def split_rows(df, dataset, seed):
    """{'train', 'val', 'test'}: row indices into df (the dataset index, sorted by path)."""
    if dataset in OFFICIAL:
        parts = {p: np.flatnonzero(df.partition.values == p) for p in ("train", "val", "test")}
        if any(len(v) == 0 for v in parts.values()):
            raise ValueError(f"{dataset}: official partition incomplete")
        return parts
    groups = df.group.tolist() if (df.group != "").any() else None
    rows = np.arange(len(df))
    fit, test = _split(rows, df.label.values, groups, seed)
    g_fit = [groups[i] for i in fit] if groups else None
    train, val = _split(fit, df.label.values[fit], g_fit, seed)
    return {"train": train, "val": val, "test": test}


def check_disjoint(df, parts):
    """Raises if a group appears in two partitions (official partitions are checked too, except the one
    BRACS patient shared by the official train and val sets, which never reaches the test set)."""
    if (df.group == "").all():
        return
    g = {k: set(df.group.values[v]) for k, v in parts.items()}
    if g["test"] & (g["train"] | g["val"]):
        raise RuntimeError(f"groups shared with the test set: {sorted(g['test'] & (g['train'] | g['val']))[:5]}")
