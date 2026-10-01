"""Uncertainty, pairwise tests and ranking stability from the probes' per-image test predictions.

For every dataset:
  * the test items of a seed are identical for all models (the split depends only on dataset and seed);
  * B cluster-bootstrap resamples per seed draw the grouping units (patient / slide / source image; the
    image itself when the dataset has none) with replacement; a resample is applied as per-item weights,
    and the SAME resample is used for every model, so model differences are paired;
  * boot[b, s, m, k] = metric k of model m on resample b of seed s. Seed-averaged replicates
    boot.mean(axis=1) give the 95% percentile CI of the seed-averaged metric and the paired test of
    every model pair (two-sided bootstrap p-value, Holm-corrected within dataset and metric).
Across datasets (primary metric per task type): Friedman test + Nemenyi critical difference on
seed-averaged means; rank stability = distribution of each model's rank over seeds x resamples.

Output (results/stats/): boot_<dataset>.npz (cache, reused only while the prediction files are unchanged), summary.csv, pairwise.csv,
ranks.csv, friedman.json.

Usage: python pipeline/stats.py [--models ...] [--datasets ...] [--n-boot 1000]
"""
import argparse
import glob
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import atomic_write_bytes, atomic_write_json  # noqa: E402
from probe import ORDINAL, RESULTS_DIR, primary  # noqa: E402
from splits import OFFICIAL  # noqa: E402

METRICS = ["acc", "bal_acc", "f1_weighted", "auc", "qwk"]
# Nemenyi critical values q_0.05 (Demsar 2006), by number of models
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def weighted_metrics(y, probs, w, n_classes):
    keep = w > 0
    y, probs, w = y[keep], probs[keep], w[keep]
    pred = probs.argmax(1)
    out = [accuracy_score(y, pred, sample_weight=w), balanced_accuracy_score(y, pred, sample_weight=w),
           f1_score(y, pred, average="weighted", sample_weight=w, zero_division=0)]
    try:
        out.append(roc_auc_score(y, probs[:, 1], sample_weight=w) if n_classes == 2 else
                   roc_auc_score(y, probs, multi_class="ovr", average="weighted", sample_weight=w,
                                 labels=list(range(n_classes))))
    except ValueError:
        out.append(np.nan)
    out.append(cohen_kappa_score(y, pred, weights="quadratic", sample_weight=w))
    return out


def _boot_seed(args):
    """All replicates of one seed: array [n_boot + 1, n_models, n_metrics]; replicate 0 = original test set.
    A resample that misses a class of the test set (AUC undefined) is redrawn; returns (out, n_redrawn)."""
    y, probs_list, groups, n_classes, n_boot, seed = args
    _, g_idx = np.unique(groups, return_inverse=True)
    n_g = g_idx.max() + 1
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot + 1, len(probs_list), len(METRICS)))
    redrawn = 0
    for b in range(n_boot + 1):
        w = np.ones(len(y))
        while b > 0:
            w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)[g_idx].astype(float)
            if len(np.unique(y[w > 0])) == len(classes):
                break
            redrawn += 1
        for m, probs in enumerate(probs_list):
            out[b, m] = weighted_metrics(y, probs, w, n_classes)
    return out, redrawn


def load_dataset(root, dataset, models):
    """{seed: (targets, groups, [probs per model])}, only seeds present for every model."""
    seeds = None
    for m in models:
        s = {int(os.path.basename(f)[4:-4]) for f in glob.glob(os.path.join(root, m, dataset, "seed*.npz"))}
        seeds = s if seeds is None else seeds & s
    data = {}
    for s in sorted(seeds or []):
        ds = [np.load(os.path.join(root, m, dataset, f"seed{s}.npz")) for m in models]
        for d in ds[1:]:
            if not (np.array_equal(d["rows"], ds[0]["rows"]) and np.array_equal(d["targets"], ds[0]["targets"])):
                raise RuntimeError(f"{dataset} seed {s}: models were evaluated on different test items")
        groups = ds[0]["groups"].astype(str)
        if (groups == "").all():
            groups = ds[0]["rows"].astype(str)  # no grouping unit: resample images
        data[s] = (ds[0]["targets"], groups, [d["probs"] for d in ds])
    if dataset in OFFICIAL and len(data) > 1:
        # Official partitions: the probe is deterministic, so every seed gives the same predictions. Keep one
        # seed; averaging bootstraps of identical predictions would shrink the CIs (by ~sqrt(n seeds)).
        first = min(data)
        for s in data:
            if not all(np.array_equal(p, q) for p, q in zip(data[s][2], data[first][2])):
                raise RuntimeError(f"{dataset}: seeds {first} and {s} differ on an official partition")
        data = {first: data[first]}
    return data


def input_hash(root, dataset, models):
    """Hash of every prediction file the bootstrap reads: rerun probes invalidate the cache."""
    import hashlib
    h = hashlib.sha256()
    for m in models:
        for f in sorted(glob.glob(os.path.join(root, m, dataset, "seed*.npz"))):
            h.update(f.encode())
            h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


def bootstrap_dataset(root, dataset, models, n_boot, out_dir, workers):
    cache = os.path.join(out_dir, f"boot_{dataset}.npz")
    key = input_hash(root, dataset, models) + ("-official1" if dataset in OFFICIAL else "") + "-allclasses"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=False)
        if list(c["models"]) == models and int(c["n_boot"]) == n_boot and str(c["input_hash"]) == key:
            return c["boot"], list(c["seeds"])
    data = load_dataset(root, dataset, models)
    if not data:
        return None, []
    n_classes = int(json.load(open(os.path.join(root, models[0], dataset, f"seed{min(data)}.json")))["n_classes"])
    jobs = [(y, p, g, n_classes, n_boot, 1000 + s) for s, (y, g, p) in data.items()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_boot_seed, jobs))
    boot = np.stack([r[0] for r in res], axis=1)  # [B+1, seeds, models, metrics]
    redrawn = np.array([r[1] for r in res])
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, boot=boot, models=np.array(models), seeds=np.array(sorted(data)), n_boot=n_boot,
                        metrics=np.array(METRICS), input_hash=key, redrawn=redrawn)
    atomic_write_bytes(cache, buf.getvalue())
    return boot, sorted(data)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="default: every model with probe results")
    ap.add_argument("--datasets", nargs="+", default=["lung", "breakhis", "nct", "hubmap", "bach", "panda", "bracs", "sicap"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--probe-dir", default=os.path.join(RESULTS_DIR, "probe"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "stats"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    models = args.models or sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.probe_dir, "*")))
    os.makedirs(args.out, exist_ok=True)

    summary, pairwise, ranks, means = [], [], [], {}
    for dataset in args.datasets:
        # only models that have this dataset, so a missing model does not block the others
        ms = [m for m in models if glob.glob(os.path.join(args.probe_dir, m, dataset, "seed*.npz"))]
        if len(ms) == 0:
            continue
        boot, seeds = bootstrap_dataset(args.probe_dir, dataset, ms, args.n_boot, args.out, args.workers)
        if boot is None:
            continue
        print(f"{dataset}: {len(ms)} models, seeds {seeds}", flush=True)
        point, reps = boot[0], boot[1:].mean(axis=1)  # [seeds, models, k], [B, models, k]
        for mi, m in enumerate(ms):
            for ki, k in enumerate(METRICS):
                lo, hi = np.nanpercentile(reps[:, mi, ki], [2.5, 97.5])
                summary.append(dict(dataset=dataset, model=m, metric=k, primary=k == primary(dataset),
                                    mean=np.nanmean(point[:, mi, ki]), sd=np.nanstd(point[:, mi, ki], ddof=1) if len(seeds) > 1 else 0.0,
                                    ci_low=lo, ci_high=hi, n_seeds=len(seeds)))
        for ki, k in enumerate(METRICS):
            rows = []
            for a, b in itertools.combinations(range(len(ms)), 2):
                d = reps[:, a, ki] - reps[:, b, ki]
                d = d[np.isfinite(d)]
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
                lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
                rows.append(dict(dataset=dataset, metric=k, model_a=ms[a], model_b=ms[b],
                                 diff=np.nanmean(point[:, a, ki] - point[:, b, ki]), ci_low=lo, ci_high=hi, p=p))
            for r, padj in zip(rows, holm([r["p"] for r in rows])):
                r["p_holm"] = padj
            pairwise += rows
        # rank stability on the primary metric: rank of each model in every seed x resample
        ki = METRICS.index(primary(dataset))
        per = boot[1:, :, :, ki].reshape(-1, len(ms))  # [(B*seeds), models]
        r = sps.rankdata(-np.nan_to_num(per, nan=-np.inf), axis=1, method="min")  # ties share the better rank
        for mi, m in enumerate(ms):
            ranks.append(dict(dataset=dataset, model=m, metric=primary(dataset), mean_rank=r[:, mi].mean(),
                              p_rank1=(r[:, mi] == 1).mean(), **{f"p_rank{j}": (r[:, mi] == j).mean() for j in range(2, len(ms) + 1)}))
        means[dataset] = {m: float(np.nanmean(point[:, mi, ki])) for mi, m in enumerate(ms)}

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(pairwise).to_csv(os.path.join(args.out, "pairwise.csv"), index=False)
    pd.DataFrame(ranks).to_csv(os.path.join(args.out, "ranks.csv"), index=False)

    # Friedman / Nemenyi over the datasets that have every model, per task type and overall
    fried = {}
    for name, dsets in [("categorical", [d for d in means if d not in ORDINAL]),
                        ("ordinal", [d for d in means if d in ORDINAL]), ("all", list(means))]:
        full = [d for d in dsets if set(means[d]) == set(models)]
        if len(full) < 2 or len(models) < 3:
            continue
        M = np.array([[means[d][m] for m in models] for d in full])
        chi2, p = sps.friedmanchisquare(*M.T)
        avg_rank = sps.rankdata(-M, axis=1).mean(0)
        k, n = len(models), len(full)
        fried[name] = {"datasets": full, "chi2": float(chi2), "p": float(p),
                       "mean_rank": dict(zip(models, map(float, avg_rank))),
                       "nemenyi_cd": float(Q05[k] * np.sqrt(k * (k + 1) / (6 * n))) if k in Q05 else None}
    atomic_write_json(os.path.join(args.out, "friedman.json"), fried)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
