"""pipeline/splits.py and pipeline/probe.py."""
import json
import os

import numpy as np
import pytest
import torch

from common import FEATURES_DIR
from data_index import load_index
from splits import check_disjoint, split_rows

import probe


@pytest.mark.parametrize("dataset", ["lung", "breakhis", "nct", "hubmap", "bach", "panda", "bracs", "sicap"])
def test_partitions_cover_and_are_disjoint(dataset):
    df = load_index(dataset)
    for seed in (42, 45):
        parts = split_rows(df, dataset, seed)
        rows = np.concatenate(list(parts.values()))
        assert len(rows) == len(set(rows)) == len(df)
        check_disjoint(df, parts)
        if (df.group != "").any() and dataset not in ("bracs",):  # BRACS official train/val share one patient
            g = {k: set(df.group.values[v]) for k, v in parts.items()}
            assert not g["train"] & g["val"]


def test_official_partitions_fixed_across_seeds():
    df = load_index("sicap")
    a, b = split_rows(df, "sicap", 42), split_rows(df, "sicap", 46)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert {k: len(v) for k, v in a.items()} == {"train": 7472, "val": 2487, "test": 2122}


@pytest.mark.parametrize("n_classes,C", [(2, 0.1), (4, 1.0), (6, 0.01)])
def test_logreg_matches_sklearn(n_classes, C):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X0 = rng.randn(600, 40).astype(np.float32)
    y = (X0[:, :n_classes] + rng.randn(600, n_classes)).argmax(1)
    X = X0 * rng.uniform(0.5, 3, 40).astype(np.float32) + rng.randn(40).astype(np.float32) * 3  # unstandardized
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours = probe.LogReg(C, device, tol=1e-9).fit(X, y, n_classes).predict_proba(X)
    # our standardization (torch.std is the sample std)
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    sk = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(Z, y).predict_proba(Z)
    assert np.abs(ours - sk).max() < 2e-3


@pytest.mark.gpu
def test_probe_end_to_end(tmp_path):
    """Real Phikon features on BreaKHis (patient-grouped): metrics sane, rerun skips, predictions consistent."""
    if not os.path.exists(os.path.join(FEATURES_DIR, "phikon", "breakhis", "DONE")):
        pytest.skip("phikon/breakhis features not extracted yet")
    import subprocess, sys
    cmd = [sys.executable, os.path.join(os.path.dirname(probe.__file__), "probe.py"), "--models", "phikon",
           "--datasets", "breakhis", "--seeds", "42", "43", "--out", str(tmp_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    for seed in (42, 43):
        r = json.load(open(tmp_path / "phikon" / "breakhis" / f"seed{seed}.json"))
        assert r["test"]["auc"] > 0.8 and r["C"] in probe.C_GRID
        assert r["counts"]["test"]["groups"] > 0
        d = np.load(tmp_path / "phikon" / "breakhis" / f"seed{seed}.npz")
        m = probe.metrics(d["targets"], d["probs"], 2)
        assert m["acc"] == pytest.approx(r["test"]["acc"])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert "[skip] phikon/breakhis" in p.stdout


def test_tissue_filter():
    from splits import usable
    df = load_index("panda")
    keep = usable("panda", df)
    assert 0.35 < 1 - keep.mean() < 0.40  # ~38% of PANDA tiles are padding
    t = __import__("pandas").read_csv(os.path.join(os.path.dirname(os.path.dirname(probe.__file__)), "indexes", "tissue_panda.csv"))
    big = t.md5.value_counts().index[:4]  # the four black/white padding patterns
    assert not keep[t.md5.isin(big).values].any()
    assert usable("nct", load_index("nct")).all()  # NCT has a real background class: never filtered
    assert usable("sicap", load_index("sicap")).all()


def test_stale_results_are_recomputed():
    """Any change of rows, labels, groups, partitions or filter changes the signature."""
    from splits import data_signature
    df = load_index("breakhis")
    sig = data_signature("breakhis", df)
    assert data_signature("breakhis", df.copy()) == sig
    for col, val in [("group", "x"), ("label", 1 - df.label.iloc[0])]:
        d = df.copy()
        d.loc[d.index[0], col] = val
        assert data_signature("breakhis", d) != sig
    assert data_signature("breakhis", df.iloc[1:]) != sig
    assert data_signature("panda", df) != sig  # same rows, different filter


def test_is_done(tmp_path):
    p = tmp_path / "seed42.json"
    p.write_text(json.dumps({"filter": {"tissue_min": 0.1}}))  # result without a signature
    assert not probe.is_done(str(p), "abc")
    p.write_text(json.dumps({"data_signature": "abc", "C_grid": probe.C_GRID}))
    assert probe.is_done(str(p), "abc") and not probe.is_done(str(p), "abd")
    p.write_text(json.dumps({"data_signature": "abc", "C_grid": [1e-4, 1.0]}))  # made with another C grid
    assert not probe.is_done(str(p), "abc")


def test_breakhis_groups_by_numeric_patient():
    df = load_index("breakhis")
    assert df.group.nunique() == 70  # not the 82 filename ids: 13412 appears as DC and LC
