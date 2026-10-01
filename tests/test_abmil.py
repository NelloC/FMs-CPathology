"""pipeline/abmil_panda.py on synthetic slides: a grade signal carried by a few tiles of each slide."""
import json

import numpy as np
import pandas as pd
import torch

import abmil_panda as ab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fake_panda(n_slides=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 6, n_slides)
    X, rows = [], []
    for s, lab in enumerate(labels):
        t = rng.integers(3, 40)
        x = rng.normal(size=(t, d)).astype(np.float32)
        k = max(1, t // 5)  # only ~20% of the tiles carry the grade
        x[:k, lab] += 3.0
        X.append(x)
        rows += [dict(path=f"s{s:04d}_{i}", group=f"s{s:04d}", label=int(lab), partition="") for i in range(t)]
    return np.concatenate(X), pd.DataFrame(rows)


def test_padding_does_not_change_predictions():
    torch.manual_seed(0)
    model = ab.GatedABMIL(8).to(DEVICE).eval()
    a = torch.randn(1, 5, 8, device=DEVICE)
    padded = torch.cat([a, torch.randn(1, 7, 8, device=DEVICE)], 1)
    mask = torch.zeros(1, 12, dtype=torch.bool, device=DEVICE)
    mask[0, :5] = True
    with torch.no_grad():
        p1 = model(a, torch.ones(1, 5, dtype=torch.bool, device=DEVICE))[0]
        p2, att = model(padded, mask)
    assert torch.allclose(p1, p2, atol=1e-6) and (att[0, 5:] == 0).all()


def test_learns_from_few_tiles_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "EPOCHS", 15)  # ~40 steps/epoch here; real PANDA has ~210
    X, df = fake_panda(n_slides=2000)
    assert ab.run_unit("fake", 42, X, df, DEVICE, str(tmp_path)) == "done"
    r = json.load(open(tmp_path / "seed42.json"))
    assert r["test"]["qwk"] > 0.8, r["test"]
    d = np.load(tmp_path / "seed42.npz")
    assert set(d.files) >= {"rows", "groups", "targets", "probs"} and d["probs"].shape[1] == 6
    # slide-disjoint and consistent with the counts
    assert r["counts"]["test"]["slides"] == len(d["targets"])
    assert ab.run_unit("fake", 42, X, df, DEVICE, str(tmp_path)) == "skip"


def test_deterministic(tmp_path):
    X, df = fake_panda(n_slides=200)
    ab.EPOCHS, old = 3, ab.EPOCHS
    try:
        ab.run_unit("fake", 43, X, df, DEVICE, str(tmp_path / "a"))
        ab.run_unit("fake", 43, X, df, DEVICE, str(tmp_path / "b"))
    finally:
        ab.EPOCHS = old
    pa, pb = np.load(tmp_path / "a" / "seed43.npz")["probs"], np.load(tmp_path / "b" / "seed43.npz")["probs"]
    assert np.allclose(pa, pb, atol=1e-5)
