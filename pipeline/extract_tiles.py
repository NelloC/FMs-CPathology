"""Fixed-magnification BRACS arm: embed the 0.5 um/px tiles (pipeline/tile_bracs.py), then mean-pool per RoI.

Step 1 (per-tile features), out: <out>/<model>/bracs_tiles/  (default features_tiles/)
  Exactly pipeline/extract.py's extract_one (same shard format, meta.json, resume, FP32 and row checks), run on the
  tile index (tiles.csv) instead of a dataset index. Each tile (224 x 224 at 0.5 um/px) goes through the model's own
  transform unchanged: Resize(shorter side -> input) + CenterCrop(input), i.e. a plain resize for square tiles
  (224 -> 448 for CONCH; Prov-GigaPath's card resizes to 256 and centre-crops 224, as in the main protocol).
  The tiling configuration hash is part of the recorded model meta, so a resume over different tiles aborts.

Step 2 (pooling), out: <pooled-out>/<model>/bracs/  (default features_bracs_tiles/)
  For every embedding key (official, cls, patch_mean when present) the RoI embedding is the mean over its tiles
  (accumulated in float64, stored float32). Written in load_features format for the BRACS index: all 4,539 RoIs in
  index order, index_hash of indexes/bracs.csv, shards/<k>.safetensors with the keys + 'row', meta.json, DONE.
  So `probe.py --features features_bracs_tiles --datasets bracs` runs unchanged.

Usage:
  python pipeline/extract_tiles.py --models phikon conch            # extract (resumable) then pool
  python pipeline/extract_tiles.py --models phikon --pool-only
  python pipeline/extract_tiles.py --models phikon --subset 50 --out /tmp/x/tiles   # smoke test (no pooling)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
from common import CODE_DIR, atomic_write_bytes, atomic_write_json, environment_info, sha256_strings, set_strict_fp32  # noqa: E402
from data_index import index_hash, load_index  # noqa: E402
from models import MODELS  # noqa: E402
from tile_bracs import OUT_DIR as TILES_DIR  # noqa: E402

TILE_DATASET = "bracs_tiles"
ROI_DATASET = "bracs"
TILE_FEATURES_DIR = os.path.join(CODE_DIR, "features_tiles")
POOLED_DIR = os.path.join(CODE_DIR, "features_bracs_tiles")


def load_tile_index(tiles_dir):
    if not os.path.exists(os.path.join(tiles_dir, "DONE")):
        raise RuntimeError(f"{tiles_dir}: tiling incomplete (run pipeline/tile_bracs.py)")
    t = pd.read_csv(os.path.join(tiles_dir, "tiles.csv"), keep_default_na=False,
                    dtype={"partition": str})
    meta = json.load(open(os.path.join(tiles_dir, "meta.json")))
    if sha256_strings(t.path.tolist()) != meta["counts"]["tile_index_hash"]:
        raise RuntimeError(f"{tiles_dir}: tiles.csv does not match meta.json")
    return t, meta


def use_tile_index(tiles_dir):
    """Make extract.load_index (used by extract_one and load_features) resolve 'bracs_tiles' to tiles.csv."""
    original = load_index

    def _load(name):
        return load_tile_index(tiles_dir)[0] if name == TILE_DATASET else original(name)

    extract.load_index = _load


def tiling_signature(tiles_meta):
    return {"tiling_config": tiles_meta["config"], "tile_index_hash": tiles_meta["counts"]["tile_index_hash"],
            "n_tiles": tiles_meta["counts"]["tiles"]}


def pool(model_name, tiles_dir, tile_root, pooled_root, shard_size=4096):
    """Mean-pool the tile features of model_name into the BRACS load_features layout. Idempotent."""
    from safetensors.numpy import load_file
    from safetensors.torch import save
    src = os.path.join(tile_root, model_name, TILE_DATASET)
    if not os.path.exists(os.path.join(src, "DONE")):
        raise RuntimeError(f"{src}: tile features incomplete")
    src_meta = json.load(open(os.path.join(src, "meta.json")))
    if src_meta["config"]["subset"]:
        raise RuntimeError(f"{src}: made with --subset; cannot pool")
    tiles, tiles_meta = load_tile_index(tiles_dir)
    df = load_index(ROI_DATASET)
    if tiles_meta["config"]["index_hash"] != index_hash(df):
        raise RuntimeError("BRACS index changed since tiling")
    if src_meta["config"]["index_hash"] != index_hash(tiles):
        raise RuntimeError(f"{src}: tile index changed since extraction")
    if src_meta["config"]["model_meta"].get("tiles") != tiling_signature(tiles_meta):
        raise RuntimeError(f"{src}: tile features were made from a different tiling")
    n = len(df)
    counts = np.bincount(tiles.roi_row.values, minlength=n)
    if len(counts) != n or (counts == 0).any():
        raise RuntimeError(f"RoIs without tiles: {np.flatnonzero(counts == 0)[:10]}")

    out_dir = os.path.join(pooled_root, model_name, ROI_DATASET)
    c = src_meta["config"]
    config = {
        "model": model_name, "dataset": ROI_DATASET, "model_meta": c["model_meta"], "transform": c["transform"],
        "index_hash": index_hash(df), "n_index_rows": n, "subset": 0, "rows_hash": index_hash(df), "n_rows": n,
        "shard_size": shard_size, "precision": c["precision"], "libs": c["libs"],
        "pooling": "mean over tiles per RoI (float64 accumulation, stored float32)",
        "tile_features": {"dir": os.path.relpath(src, CODE_DIR), "index_hash": c["index_hash"], "n_rows": c["n_rows"],
                          "shard_size": c["shard_size"]},
        "tiling": tiling_signature(tiles_meta),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))["config"]
        if old != config:
            diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
            raise RuntimeError(f"{out_dir}: existing pooled features have a different configuration: {diff}")
        if os.path.exists(os.path.join(out_dir, "DONE")):
            print(f"[skip] pooled {model_name}: complete", flush=True)
            return out_dir
    else:
        atomic_write_json(meta_path, {"config": config, "environment": src_meta.get("environment"),
                                      "tiling_resolution_evidence": tiles_meta.get("resolution_evidence"),
                                      "tiles_per_roi": tiles_meta["counts"]["tiles_per_roi"],
                                      "started": time.strftime("%Y-%m-%d %H:%M:%S")})

    sums, seen = {}, 0
    n_src = (c["n_rows"] + c["shard_size"] - 1) // c["shard_size"]
    for k in range(n_src):
        s = load_file(extract.shard_path(src, k))
        rows = s.pop("row")
        if not np.array_equal(rows, np.arange(seen, seen + len(rows))):
            raise RuntimeError(f"{src} shard {k}: unexpected rows")
        seen += len(rows)
        roi = tiles.roi_row.values[rows]
        for key, v in s.items():
            acc = sums.setdefault(key, np.zeros((n, v.shape[1]), np.float64))
            np.add.at(acc, roi, v.astype(np.float64))
    if seen != len(tiles):
        raise RuntimeError(f"{src}: {seen} tile rows, expected {len(tiles)}")
    means = {key: (v / counts[:, None]).astype(np.float32) for key, v in sums.items()}
    for key, v in means.items():
        if not np.isfinite(v).all():
            raise RuntimeError(f"pooled {key}: non-finite values")
    for k in range((n + shard_size - 1) // shard_size):
        lo, hi = k * shard_size, min(n, (k + 1) * shard_size)
        tensors = {key: torch.from_numpy(np.ascontiguousarray(v[lo:hi])) for key, v in means.items()}
        tensors["row"] = torch.arange(lo, hi, dtype=torch.int64)
        atomic_write_bytes(extract.shard_path(out_dir, k), save(tensors, metadata={
            "model": model_name, "dataset": ROI_DATASET, "shard": str(k), "pooling": "mean"}))
    atomic_write_bytes(os.path.join(out_dir, "DONE"), time.strftime("%Y-%m-%d %H:%M:%S\n").encode())
    print(f"[done] pooled {model_name}: {n} RoIs from {len(tiles)} tiles, keys {sorted(means)}", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--tiles-dir", default=TILES_DIR)
    ap.add_argument("--out", default=TILE_FEATURES_DIR, help="per-tile features root")
    ap.add_argument("--pooled-out", default=POOLED_DIR, help="pooled per-RoI features root (load_features format)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--subset", type=int, default=0, help="evenly spaced subset of N tiles (smoke tests; no pooling)")
    ap.add_argument("--pool-only", action="store_true")
    args = ap.parse_args()
    if args.subset and os.path.abspath(args.out) == os.path.abspath(TILE_FEATURES_DIR):
        ap.error("--subset requires a separate --out directory")

    use_tile_index(args.tiles_dir)
    _, tiles_meta = load_tile_index(args.tiles_dir)
    failures = []
    if not args.pool_only:
        set_strict_fp32()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = environment_info()
    for model_name in args.models:
        try:
            if not args.pool_only:
                done = os.path.exists(os.path.join(args.out, model_name, TILE_DATASET, "DONE"))
                if not done:
                    extractor, transform, meta_model = MODELS[model_name]()
                    meta_model = {**meta_model, "tiles": tiling_signature(tiles_meta)}
                    extractor = extractor.to(device)
                    extract.extract_one(model_name, TILE_DATASET, extractor, transform, meta_model, args, device, env)
                    del extractor
                    torch.cuda.empty_cache()
            if not args.subset:
                pool(model_name, args.tiles_dir, args.out, args.pooled_out)
        except Exception as e:  # one failing model must not stop the others; a rerun retries it
            failures.append(f"{model_name}: {e!r}")
            print(f"[FAILED] {failures[-1]}", flush=True)
            torch.cuda.empty_cache()
    if failures:
        print(f"{len(failures)} model(s) failed:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
