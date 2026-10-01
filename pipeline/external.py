"""External test: NCT-CRC-HE-100K probes evaluated on CRC-VAL-HE-7K.

CRC-VAL-HE-7K (Kather et al., Zenodo 1214456) comes from patients not in NCT-CRC-HE-100K, so it is the
independent test set that the patch-level NCT split cannot provide. For every model and seed, the probe is
refitted exactly as in probe.py (same train + val rows of that seed, same C chosen on the validation part,
read from results/probe/<model>/nct/seed<k>.json) and applied unchanged to all CRC-VAL-HE-7K images.
Both datasets list the same nine class folders, so labels coincide (checked).

Output: results/external/<model>/nct_val7k/seed<k>.{json,npz}; the .npz has the fields stats.py reads, so
  python pipeline/stats.py --probe-dir results/external --out results/stats_external --datasets nct_val7k
gives CIs and paired tests (image-level bootstrap: the external set has no patient identifiers).
A unit whose .json was made from the same probe result and external index is skipped.

Usage: python pipeline/external.py --models phikon uni2h ...
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_DIR, atomic_write_bytes, atomic_write_json, environment_info, set_strict_fp32  # noqa: E402
from probe import RESULTS_DIR, SEEDS, LogReg, metrics  # noqa: E402
from splits import split_rows  # noqa: E402


def signature(probe_json, ext_df):
    h = hashlib.sha256(json.dumps(probe_json.get("data_signature")).encode() + repr(probe_json["C"]).encode())
    for col in ("path", "label"):
        h.update("\n".join(map(str, ext_df[col].values)).encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--features", default=FEATURES_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "external"))
    args = ap.parse_args()
    from extract import load_features
    set_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = []
    for model in args.models:
        try:
            X, df = load_features(model, "nct", root=args.features)
            Xe, dfe = load_features(model, "nct_val7k", root=args.features)
        except RuntimeError as e:
            print(f"[wait] {model}: {e}", flush=True)
            continue
        cls, cls_e = dict(zip(df.label, df.class_name)), dict(zip(dfe.label, dfe.class_name))
        if cls != cls_e:
            raise RuntimeError(f"class mapping differs: {cls} vs {cls_e}")
        out_dir = os.path.join(args.out, model, "nct_val7k")
        os.makedirs(out_dir, exist_ok=True)
        atomic_write_json(os.path.join(out_dir, "meta.json"), {"environment": environment_info()})
        y, ye = df.label.values, dfe.label.values
        for seed in args.seeds:
            pj_path = os.path.join(RESULTS_DIR, "probe", model, "nct", f"seed{seed}.json")
            if not os.path.exists(pj_path):
                print(f"[wait] {model} seed {seed}: no NCT probe result", flush=True)
                continue
            pj = json.load(open(pj_path))
            sig = signature(pj, dfe)
            json_path = os.path.join(out_dir, f"seed{seed}.json")
            if os.path.exists(json_path) and json.load(open(json_path)).get("signature") == sig:
                continue
            try:
                t0 = time.time()
                parts = split_rows(df, "nct", seed)
                fit = np.concatenate([parts["train"], parts["val"]])
                k = int(y.max()) + 1
                lr = LogReg(pj["C"], device).fit(X[fit], y[fit], k)
                probs = lr.predict_proba(Xe)
                internal = metrics(y[parts["test"]], lr.predict_proba(X[parts["test"]]), k)
                buf = io.BytesIO()
                np.savez_compressed(buf, rows=np.arange(len(dfe)), paths=np.asarray(dfe.path.to_numpy(), dtype=str),
                                    groups=np.full(len(dfe), "", dtype=str), targets=ye, probs=probs.astype(np.float32))
                atomic_write_bytes(os.path.join(out_dir, f"seed{seed}.npz"), buf.getvalue())
                r = {"model": model, "dataset": "nct_val7k", "trained_on": "nct", "seed": seed, "C": pj["C"],
                     "signature": sig, "n_train": int(len(fit)), "n_external": int(len(dfe)),
                     "test": metrics(ye, probs, k), "internal_test": internal,
                     "internal_test_matches_probe": bool(abs(internal["auc"] - pj["test"]["auc"]) < 1e-6),
                     "primary_metric": "auc", "task_type": "categorical", "n_classes": k, "seconds": round(time.time() - t0, 1)}
                atomic_write_json(json_path, r)
                print(f"{model:10s} seed {seed} | external acc {r['test']['acc']:.4f} auc {r['test']['auc']:.4f} "
                      f"| internal auc {internal['auc']:.4f} (probe {pj['test']['auc']:.4f})", flush=True)
            except Exception as e:  # one failing unit must not stop the others
                failures.append(f"{model}/seed{seed}: {e!r}")
                print(f"[FAILED] {failures[-1]}", flush=True)
    if failures:
        print(f"{len(failures)} unit(s) failed:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
