"""Integrity check of every finished feature set (features/<model>/<dataset>/DONE).

Per set: configuration matches the current index (same files, same order); every index row present
exactly once and in order; all values finite; no all-zero rows; number of exact-duplicate embeddings
(many would indicate failed reads) compared with duplicate image files; and K random images re-embedded
from scratch with the model's official loader must match the cached rows.

Output: verification/features.json. Usage: python pipeline/verify_features.py [--k 16]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, CODE_DIR, FEATURES_DIR, atomic_write_json, set_strict_fp32  # noqa: E402
from data_index import index_hash, load_index  # noqa: E402
from extract import WHOLE_IMAGE, shard_path, whole_image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def check_set(model, dataset, extractor, transform, k, device):
    out_dir = os.path.join(FEATURES_DIR, model, dataset)
    cfg = json.load(open(os.path.join(out_dir, "meta.json")))["config"]
    df = load_index(dataset)
    r = {"n_index": len(df), "index_hash_ok": cfg["index_hash"] == index_hash(df), "subset": cfg["subset"]}
    from safetensors.numpy import load_file
    n_shards = (cfg["n_rows"] + cfg["shard_size"] - 1) // cfg["shard_size"]
    rows, feats = [], {}
    for s in range(n_shards):
        t = load_file(shard_path(out_dir, s))
        rows.append(t.pop("row"))
        for key, v in t.items():
            feats.setdefault(key, []).append(v)
    rows = np.concatenate(rows)
    feats = {key: np.concatenate(v) for key, v in feats.items()}
    X = feats["official"]
    r["rows_complete_and_ordered"] = bool(np.array_equal(rows, np.arange(len(df))))
    r["keys"] = {key: list(v.shape) for key, v in feats.items()}
    r["all_finite"] = bool(all(np.isfinite(v).all() for v in feats.values()))
    r["zero_rows"] = int((np.abs(X).sum(1) == 0).sum())
    uniq = np.unique(X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))), return_counts=True)[1]
    r["duplicate_embeddings"] = int((uniq - 1).sum())
    # re-embed K random images from scratch, with the same transform as extract.py
    if dataset in WHOLE_IMAGE:
        transform = whole_image(transform)
    pick = np.sort(np.random.RandomState(0).choice(len(df), min(k, len(df)), replace=False))
    x = torch.stack([transform(Image.open(os.path.join(BASE_DIR, df.path.values[i])).convert("RGB")) for i in pick])
    with torch.no_grad():
        e = extractor(x.to(device))["official"].float().cpu().numpy()
    diff = np.abs(e - X[pick]).max()
    r["recompute_max_abs_diff"] = float(diff)
    r["recompute_rel_diff"] = float(diff / np.abs(X[pick]).max())
    r["ok"] = (r["index_hash_ok"] and r["rows_complete_and_ordered"] and r["all_finite"] and r["zero_rows"] == 0
               and r["recompute_rel_diff"] < 1e-4 and not cfg["subset"])
    return r


def duplicate_files(dataset, df):
    """Exact duplicate image files (by content hash) in the dataset, to compare with duplicate embeddings."""
    seen, dup = set(), 0
    for p in df.path.values:
        h = hashlib.md5(open(os.path.join(BASE_DIR, p), "rb").read()).digest()
        dup += h in seen
        seen.add(h)
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    from models import MODELS
    set_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_path = os.path.join(CODE_DIR, "verification", "features.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    dup_files = report.get("_duplicate_files", {})
    for model in args.models or sorted(os.listdir(FEATURES_DIR)):
        done = sorted(d for d in os.listdir(os.path.join(FEATURES_DIR, model))
                      if os.path.exists(os.path.join(FEATURES_DIR, model, d, "DONE")))
        todo = [d for d in done if not report.get(model, {}).get(d, {}).get("ok")]
        if not todo:
            continue
        extractor, transform, _ = MODELS[model]()
        extractor = extractor.to(device)
        for dataset in todo:
            if dataset not in dup_files:
                dup_files[dataset] = duplicate_files(dataset, load_index(dataset))
            r = check_set(model, dataset, extractor, transform, args.k, device)
            r["duplicate_image_files"] = dup_files[dataset]
            report.setdefault(model, {})[dataset] = r
            report["_duplicate_files"] = dup_files
            atomic_write_json(report_path, report)
            print(f"{model:10s} {dataset:9s} ok={r['ok']} rel_diff={r['recompute_rel_diff']:.1e} "
                  f"dup_emb={r['duplicate_embeddings']} dup_files={r['duplicate_image_files']}", flush=True)
        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
