"""Linear probes on cached frozen embeddings: one unit per model x dataset x seed. Resumable.

Probe: L2-regularized logistic regression (binary or multinomial) on standardized features (mean/std of the
fitting set), the objective of sklearn's LogisticRegression (C * sum of cross-entropies + ||W||^2 / 2,
bias not penalized), solved by full-batch L-BFGS on the GPU. C is chosen on the validation part
(primary metric: AUC for categorical tasks, QWK for ordinal ones; ties -> smaller C), then the probe
is refitted on train + val and evaluated once on the test part. The probe is convex and deterministic:
with official partitions (BRACS, SICAPv2) every seed gives the same result, and uncertainty comes
from the cluster bootstrap in stats.py.

Output: results/probe/<model>/<dataset>/seed<k>.json (metrics, chosen C, counts) and .npz
(per-image test predictions: index rows, paths, groups, targets, probs). A unit whose .json was made from the
same input (splits.data_signature) and the same C grid is skipped.

Usage: python pipeline/probe.py --models phikon uni2h --datasets nct bach --seeds 42 43 44 45 46
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CODE_DIR, FEATURES_DIR, atomic_write_bytes, atomic_write_json, environment_info, set_strict_fp32  # noqa: E402
from splits import OFFICIAL, check_disjoint, data_signature, filter_config, split_rows, usable  # noqa: E402

RESULTS_DIR = os.environ.get("VFM_RESULTS_DIR", os.path.join(CODE_DIR, "results"))
ORDINAL = {"panda", "bracs", "sicap"}
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SEEDS = [42, 43, 44, 45, 46]
MAX_ITER = 5000  # L-BFGS iterations per fit


def metrics(y, probs, n_classes):
    pred = probs.argmax(1)
    out = {"acc": float(accuracy_score(y, pred)), "bal_acc": float(balanced_accuracy_score(y, pred)),
           "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
           "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
           "qwk": float(cohen_kappa_score(y, pred, weights="quadratic"))}
    try:
        out["auc"] = float(roc_auc_score(y, probs[:, 1]) if n_classes == 2 else
                           roc_auc_score(y, probs, multi_class="ovr", average="weighted", labels=list(range(n_classes))))
    except ValueError:
        out["auc"] = float("nan")
    return out


def primary(dataset):
    return "qwk" if dataset in ORDINAL else "auc"


class LogReg:
    """sklearn-equivalent logistic regression (binary: one logit; otherwise multinomial), full-batch L-BFGS
    in float32 on `device`."""

    def __init__(self, C, device, max_iter=MAX_ITER, tol=1e-6):
        self.C, self.device, self.max_iter, self.tol = C, device, max_iter, tol

    def fit(self, X, y, n_classes):
        X = torch.as_tensor(X, device=self.device)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Z = (X - self.mean) / self.std
        y = torch.as_tensor(y, device=self.device, dtype=torch.long)
        n, d = Z.shape
        # binary: a single logit against a fixed 0, so the penalty is sklearn's ||w||^2 / 2 (two softmax
        # columns would halve it)
        k = 1 if n_classes == 2 else n_classes
        W = torch.zeros(d, k, device=self.device, requires_grad=True)
        b = torch.zeros(k, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")
        reg = 1.0 / (2 * self.C * n)  # sklearn objective divided by C*n

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self._logits(Z, W, b), y) + reg * (W * W).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        self.n_iter = opt.state[opt._params[0]]["n_iter"]
        return self

    @staticmethod
    def _logits(Z, W, b):
        out = Z @ W + b
        return torch.cat([torch.zeros_like(out), out], 1) if out.shape[1] == 1 else out

    @torch.no_grad()
    def predict_proba(self, X):
        Z = (torch.as_tensor(X, device=self.device) - self.mean) / self.std
        return torch.softmax(self._logits(Z, self.W, self.b), 1).cpu().numpy()


def is_done(json_path, signature, select="auc", split="grouped"):
    """A unit counts as done only if it was computed from exactly the current input (data_signature), C grid and
    protocol variant (C selection criterion, split type), and no fit stopped at a lower iteration cap than MAX_ITER."""
    if not os.path.exists(json_path):
        return False
    r = json.load(open(json_path))
    cap = r.get("max_iter", 1000)
    iters = [v["n_iter"] for v in r.get("sweep", {}).values()] + [r.get("n_iter", 0)]
    return (r.get("data_signature") == signature and r.get("C_grid") == C_GRID
            and r.get("selection", "auc") == select and r.get("split", "grouped") == split
            and (cap >= MAX_ITER or max(iters) < cap))


def variant_dir(out, key="official", select="auc", split="grouped"):
    """Default protocol -> results/probe; variants get suffixes (_cls, _patch_mean, _logloss, _imagesplit)."""
    return out + (f"_{key}" if key != "official" else "") + ("_logloss" if select == "logloss" else "") \
        + ("_imagesplit" if split == "image" else "")


def run_unit(model, dataset, seed, X, df, device, out_dir, select="auc", split="grouped"):
    """X, df: usable rows only; df keeps the original index row numbers (saved as 'rows').
    select: C chosen by validation primary metric ('auc': AUC / QWK) or by validation log-loss ('logloss').
    split: 'grouped' (default) or 'image' (groups ignored: image-level stratified split, to measure leakage)."""
    json_path = os.path.join(out_dir, f"seed{seed}.json")
    signature = data_signature(dataset, df)
    if is_done(json_path, signature, select, split):
        return "skip"
    t0 = time.time()
    if split == "image":
        if dataset in OFFICIAL:
            raise ValueError("image-level split is meaningless for official partitions")
        parts = split_rows(df.assign(group=""), dataset, seed)
    else:
        parts = split_rows(df, dataset, seed)
        check_disjoint(df, parts)
    y = df.label.values
    k = int(y.max()) + 1
    key = primary(dataset)

    # choose C on val; every C is solved from zero, independently of the others
    sweep = {}
    for C in C_GRID:
        lr = LogReg(C, device).fit(X[parts["train"]], y[parts["train"]], k)
        pv = lr.predict_proba(X[parts["val"]])
        m = metrics(y[parts["val"]], pv, k)
        m["log_loss"] = float(log_loss(y[parts["val"]], np.clip(pv, 1e-12, 1), labels=list(range(k))))
        sweep[str(C)] = {"val": m, "n_iter": lr.n_iter}
    if select == "logloss":
        best = min(C_GRID, key=lambda C: (np.nan_to_num(sweep[str(C)]["val"]["log_loss"], nan=np.inf), C))
    else:
        best = max(C_GRID, key=lambda C: (np.nan_to_num(sweep[str(C)]["val"][key], nan=-np.inf), -C))

    fit = np.concatenate([parts["train"], parts["val"]])
    lr = LogReg(best, device).fit(X[fit], y[fit], k)
    probs = lr.predict_proba(X[parts["test"]])
    test = parts["test"]

    buf = io.BytesIO()
    np.savez_compressed(buf, rows=df.index.values[test], paths=np.asarray(df.path.to_numpy()[test], dtype=str),
                        groups=np.asarray(df.group.to_numpy()[test], dtype=str),
                        targets=y[test], probs=probs.astype(np.float32))
    atomic_write_bytes(os.path.join(out_dir, f"seed{seed}.npz"), buf.getvalue())
    atomic_write_json(json_path, {
        "model": model, "dataset": dataset, "seed": seed, "task_type": "ordinal" if dataset in ORDINAL else "categorical",
        "filter": filter_config(dataset), "data_signature": signature, "n_index_rows_used": int(len(df)),
        "primary_metric": key, "selection": select, "split": split,
        "C": best, "C_grid": C_GRID, "sweep": sweep, "n_iter": lr.n_iter, "max_iter": MAX_ITER,
        "test": metrics(y[test], probs, k), "n_classes": k,
        "counts": {p: {"images": int(len(v)), "groups": int(len(set(df.group.values[v]))) if (df.group != "").any() else None}
                   for p, v in parts.items()},
        "seconds": round(time.time() - t0, 1),
    })
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--key", default="official", help="embedding type (official, cls, patch_mean)")
    ap.add_argument("--features", default=FEATURES_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "probe"))
    ap.add_argument("--select", default="auc", choices=["auc", "logloss"],
                    help="C selection: validation primary metric (default) or validation log-loss")
    ap.add_argument("--split", default="grouped", choices=["grouped", "image"],
                    help="grouped (default) or image-level split ignoring groups (leakage measurement)")
    args = ap.parse_args()

    from extract import load_features
    set_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = []
    for model in args.models:
        for dataset in args.datasets:
            out_dir = os.path.join(variant_dir(args.out, args.key, args.select, args.split), model, dataset)
            from data_index import load_index
            idx = load_index(dataset)
            signature = data_signature(dataset, idx[usable(dataset, idx)])
            if all(is_done(os.path.join(out_dir, f"seed{s}.json"), signature, args.select, args.split)
                   for s in args.seeds):
                print(f"[skip] {model}/{dataset}", flush=True)
                continue
            try:
                X, df = load_features(model, dataset, key=args.key, root=args.features)
            except RuntimeError as e:  # extraction not finished yet
                print(f"[wait] {model}/{dataset}: {e}", flush=True)
                continue
            keep = usable(dataset, df)
            X, df = X[keep], df[keep]  # df keeps the original row numbers as its index
            os.makedirs(out_dir, exist_ok=True)
            meta = os.path.join(out_dir, "meta.json")
            atomic_write_json(meta, {"features": json.load(open(os.path.join(args.features, model, dataset, "meta.json"))),
                                     "embedding_key": args.key, "selection": args.select, "split": args.split,
                                     "environment": environment_info()})
            for seed in args.seeds:
                try:
                    status = run_unit(model, dataset, seed, X, df, device, out_dir, args.select, args.split)
                except Exception as e:  # one failing unit must not stop the others; a rerun retries it
                    failures.append(f"{model}/{dataset}/seed{seed}: {e!r}")
                    print(f"[FAILED] {failures[-1]}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                if status == "done":
                    r = json.load(open(os.path.join(out_dir, f"seed{seed}.json")))
                    t = r["test"]
                    print(f"{model:10s} {dataset:9s} seed {seed} | C={r['C']:g} | acc {t['acc']:.4f} auc {t['auc']:.4f} "
                          f"qwk {t['qwk']:.4f} | {r['seconds']}s", flush=True)
    if failures:
        print(f"{len(failures)} unit(s) failed:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
