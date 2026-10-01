"""Extract frozen embeddings once per model x dataset, in strict FP32. Resumable.

Output: features/<model>/<dataset>/
  meta.json               configuration + environment; a restart with a different configuration aborts
  shards/<k>.safetensors  rows [k*shard_size, (k+1)*shard_size) of the index, one tensor per embedding
                          type plus 'row' (index row numbers); written atomically
  DONE                    written when every shard exists

A crash loses at most the shard in progress. Restarting the same command skips finished shards and
produces identical features (no randomness; batches never cross shard boundaries).

Usage:
  python pipeline/extract.py --models uni2h virchow2 --datasets panda bracs
  python pipeline/extract.py --models phikon --datasets nct --subset 512 --out /tmp/smoke   # smoke test
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, FEATURES_DIR, atomic_write_bytes, atomic_write_json, environment_info, set_strict_fp32
from data_index import index_hash, load_index
from models import MODELS

Image.MAX_IMAGE_PIXELS = None


class ImageRows(Dataset):
    def __init__(self, paths, rows, transform):
        self.paths, self.rows, self.transform = paths, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = os.path.join(BASE_DIR, self.paths[row])
        try:
            with Image.open(path) as img:
                x = self.transform(img.convert("RGB"))
        except Exception as e:  # fail loudly: no silent black images
            raise RuntimeError(f"cannot read {path}: {e}") from e
        return x, row


# Datasets whose images are whole regions of interest of varying size and aspect ratio (BRACS): each RoI is
# resized as a whole to a square before the model's own transform, so nothing is cropped away (the model's
# Resize is then a no-op and its CenterCrop only acts where the model card itself crops, i.e. Prov-GigaPath).
WHOLE_IMAGE = {"bracs"}


def whole_image(transform):
    from torchvision import transforms as T
    first = transform.transforms[0]
    if not isinstance(first, T.Resize) or not isinstance(first.size, (int, list, tuple)):
        raise RuntimeError(f"unexpected transform for whole-image resize: {transform!r}")
    s = first.size if isinstance(first.size, int) else first.size[0]
    return T.Compose([T.Resize((s, s), interpolation=first.interpolation), *transform.transforms])


def shard_path(out_dir, k):
    return os.path.join(out_dir, "shards", f"{k:05d}.safetensors")


def extract_one(model_name, dataset, extractor, transform, meta_model, args, device, env):
    from safetensors.torch import save
    if dataset in WHOLE_IMAGE:
        transform = whole_image(transform)
    df = load_index(dataset)
    rows_all = np.arange(len(df))
    if args.subset:
        rows_all = np.unique(np.linspace(0, len(df) - 1, args.subset).round().astype(int))
    out_dir = os.path.join(args.out, model_name, dataset)
    config = {
        "model": model_name, "dataset": dataset, "model_meta": meta_model, "transform": repr(transform),
        "index_hash": index_hash(df), "n_index_rows": len(df), "subset": args.subset,
        "rows_hash": index_hash(df.iloc[rows_all]), "n_rows": len(rows_all),
        "shard_size": args.shard_size, "precision": "fp32 (TF32 disabled)",
        # Library versions that can change embeddings: resuming under different ones aborts.
        "libs": {k: env.get(k) for k in ["torch", "torchvision", "timm", "transformers", "conch", "pillow",
                                         "cuda", "cudnn", "gpu"]},
    }
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        import json
        old = json.load(open(meta_path))
        if old["config"] != config:
            diff = {k: (old["config"].get(k), v) for k, v in config.items() if old["config"].get(k) != v}
            raise RuntimeError(f"{out_dir}: existing features were made with a different configuration: {diff}")
    else:
        atomic_write_json(meta_path, {"config": config, "environment": env, "batch_size": args.batch_size,
                                      "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print(f"[skip] {model_name}/{dataset}: complete", flush=True)
        return

    n_shards = (len(rows_all) + args.shard_size - 1) // args.shard_size
    pending = [k for k in range(n_shards) if not os.path.exists(shard_path(out_dir, k))]
    print(f"[run] {model_name}/{dataset}: {len(rows_all)} images, {n_shards} shards, {len(pending)} pending",
          flush=True)
    paths = df.path.tolist()
    t0, done_imgs = time.time(), 0
    for k in pending:
        rows = rows_all[k * args.shard_size:(k + 1) * args.shard_size].tolist()
        loader = DataLoader(ImageRows(paths, rows, transform), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, prefetch_factor=4 if args.workers else None)
        feats, got_rows = {}, []
        with torch.inference_mode():
            for x, r in loader:
                out = extractor(x.to(device, non_blocking=True))
                for name, v in out.items():
                    if v.dtype != torch.float32:
                        raise RuntimeError(f"{name} is {v.dtype}, expected float32")
                    feats.setdefault(name, []).append(v.cpu())
                got_rows.append(r)
        tensors = {name: torch.cat(v).contiguous() for name, v in feats.items()}
        tensors["row"] = torch.cat(got_rows).to(torch.int64)
        if tensors["row"].tolist() != rows:
            raise RuntimeError(f"shard {k}: row order mismatch")
        for name, v in tensors.items():
            if name != "row" and not torch.isfinite(v).all():
                raise RuntimeError(f"shard {k}: non-finite values in {name}")
        atomic_write_bytes(shard_path(out_dir, k), save(tensors, metadata={
            "model": model_name, "dataset": dataset, "shard": str(k), "batch_size": str(args.batch_size)}))
        done_imgs += len(rows)
        rate = done_imgs / (time.time() - t0)
        left = sum(min(args.shard_size, len(rows_all) - j * args.shard_size) for j in pending if j > k)
        print(f"  shard {k + 1}/{n_shards} | {rate:.1f} img/s | ETA {left / rate / 3600:.2f} h", flush=True)

    if all(os.path.exists(shard_path(out_dir, k)) for k in range(n_shards)):
        atomic_write_bytes(os.path.join(out_dir, "DONE"), time.strftime("%Y-%m-%d %H:%M:%S\n").encode())
        print(f"[done] {model_name}/{dataset}", flush=True)


def load_features(model_name, dataset, key="official", root=FEATURES_DIR):
    """Return (features [n, d] float32 array, index DataFrame aligned row by row). Verifies completeness."""
    import json
    from safetensors.numpy import load_file
    out_dir = os.path.join(root, model_name, dataset)
    if not os.path.exists(os.path.join(out_dir, "DONE")):
        raise RuntimeError(f"{out_dir} is incomplete")
    config = json.load(open(os.path.join(out_dir, "meta.json")))["config"]
    df = load_index(dataset)
    if index_hash(df) != config["index_hash"]:
        raise RuntimeError(f"{dataset}: index changed since extraction")
    n_shards = (config["n_rows"] + config["shard_size"] - 1) // config["shard_size"]
    parts, rows = [], []
    for k in range(n_shards):
        s = load_file(shard_path(out_dir, k))
        parts.append(s[key])
        rows.append(s["row"])
    rows = np.concatenate(rows)
    return np.concatenate(parts), df.iloc[rows].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--subset", type=int, default=0, help="evenly spaced subset of N rows (smoke tests only)")
    ap.add_argument("--out", default=FEATURES_DIR)
    args = ap.parse_args()
    if args.subset and os.path.abspath(args.out) == os.path.abspath(FEATURES_DIR):
        ap.error("--subset requires a separate --out directory")

    set_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment_info()
    for model_name in args.models:
        extractor, transform, meta_model = MODELS[model_name]()
        extractor = extractor.to(device)
        for dataset in args.datasets:
            extract_one(model_name, dataset, extractor, transform, meta_model, args, device, env)
        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
