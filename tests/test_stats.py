"""pipeline/stats.py on synthetic probe outputs with known answers."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import stats
from conftest import PIPELINE_DIR


def test_holm():
    # hand-computed: p sorted .01,.02,.04 -> .03,.04,.04 (monotone)
    assert np.allclose(stats.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def _write(root, model, dataset, seed, rows, groups, y, probs):
    d = os.path.join(root, model, dataset)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"seed{seed}.npz"), rows=rows, paths=rows.astype(str), groups=groups, targets=y,
             probs=probs.astype(np.float32))
    json.dump({"n_classes": probs.shape[1]}, open(os.path.join(d, f"seed{seed}.json"), "w"))


@pytest.fixture(scope="module")
def run_stats(tmp_path_factory):
    """3 models x 2 datasets x 2 seeds: 'good' separates classes, 'good_copy' is identical to it, 'noise' is random.
    Dataset 'clus' has 20 groups of 25 identical items (strong clustering); 'flat' has no groups."""
    root = tmp_path_factory.mktemp("probe")
    rng = np.random.default_rng(0)
    for dataset in ("clus", "flat"):
        for seed in (42, 43):
            n = 500
            y = np.repeat(rng.integers(0, 2, 20), 25) if dataset == "clus" else rng.integers(0, 2, n)
            groups = np.repeat(np.arange(20), 25).astype(str) if dataset == "clus" else np.array([""] * n)
            score = np.clip(y * 0.6 + rng.random(n) * 0.5, 0, 1)
            good = np.stack([1 - score, score], 1)
            noise = rng.dirichlet([1, 1], n)
            if dataset == "clus":  # items of a cluster share their prediction: fully correlated within groups
                noise = np.repeat(noise[::25], 25, axis=0)
            rows = np.arange(n) + 1000 * seed
            _write(root, "good", dataset, seed, rows, groups, y, good)
            _write(root, "good_copy", dataset, seed, rows, groups, y, good)
            _write(root, "noise", dataset, seed, rows, groups, y, noise)
    out = tmp_path_factory.mktemp("stats")
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, "stats.py"), "--probe-dir", str(root), "--out", str(out),
           "--datasets", "clus", "flat", "--n-boot", "300", "--workers", "4"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return out, cmd


def test_pairwise(run_stats):
    pw = pd.read_csv(run_stats[0] / "pairwise.csv")
    auc = pw[pw.metric == "auc"].set_index(["dataset", "model_a", "model_b"])
    for d in ("clus", "flat"):
        same = auc.loc[(d, "good", "good_copy")]
        assert same["diff"] == 0 and same["p_holm"] == 1.0
        assert auc.loc[(d, "good", "noise")]["p_holm"] < 0.05


def test_cluster_bootstrap_widens_ci(run_stats):
    s = pd.read_csv(run_stats[0] / "summary.csv")
    s = s[(s.model == "noise") & (s.metric == "acc")].set_index("dataset")
    # 20 clusters of 25 identical items behave like n=20, not n=500
    assert (s.ci_high - s.ci_low)["clus"] > 2 * (s.ci_high - s.ci_low)["flat"]


def test_ranks_and_rerun(run_stats):
    r = pd.read_csv(run_stats[0] / "ranks.csv")
    noise, good = r[r.model == "noise"], r[r.model == "good"]
    assert (good.p_rank1 == 1).all()  # tied with its copy: both rank 1
    assert (noise.mean_rank > 2.9).all() and (noise.p_rank1 < 0.05).all()
    # rerun uses the cached bootstrap and reproduces the outputs
    before = (run_stats[0] / "summary.csv").read_text()
    assert subprocess.run(run_stats[1], capture_output=True).returncode == 0
    assert (run_stats[0] / "summary.csv").read_text() == before


def test_cache_invalidated_when_predictions_change(tmp_path):
    root = tmp_path / "p"
    y = np.array([0, 1] * 20)
    probs = np.stack([1 - y * .8, y * .8 + .1], 1)
    for m in ("a", "b"):
        _write(root, m, "d", 42, np.arange(40), np.array([""] * 40), y, probs)
    k1 = stats.input_hash(str(root), "d", ["a", "b"])
    _write(root, "b", "d", 42, np.arange(40), np.array([""] * 40), y, probs[:, ::-1])  # probe rerun
    assert stats.input_hash(str(root), "d", ["a", "b"]) != k1
