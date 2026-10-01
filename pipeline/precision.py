"""Numerical precision and latency on one host: FP32 / FP16 / BF16 on the GPU, FP32 / INT8 on the CPU.

For every model x categorical dataset, a fixed subset of the seed-42 test images (--n-images, drawn with
seed 42, identical for every model and precision) is embedded at each precision. The classifier is the
seed-42 linear probe, refitted from the cached FP32 features with the C chosen by probe.py (the probe is
deterministic), and applied in FP32 to each precision's features. Reported per precision: probe metrics,
agreement of predictions and cosine similarity of the features with FP32 GPU (INT8 is compared with FP32 CPU
in report.py), and backbone latency.

Timing: batch 32; the subset's full batches are held in memory (data loading excluded); 3 warm-up batches;
the full batches are timed 3 times; mean and SD of ms/image and images/s; CPU thread count recorded.
INT8 = torch.ao dynamic quantization of nn.Linear (weights INT8, activations quantized at run time),
qnnpack on ARM / fbgemm on x86; convolutions (CTransPath stem and stages, patch embeddings) stay FP32.
Run it when nothing else uses the GPU or the CPU, or the timings are meaningless.

Output: results/precision/<model>/<dataset>/<precision>.{json,npz}; a unit whose .json exists is skipped.
Usage: python pipeline/precision.py --models phikon uni2h --datasets nct bach [--n-images 500]
"""
import argparse
import copy
import io
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, FEATURES_DIR, atomic_write_bytes, atomic_write_json, environment_info, set_strict_fp32  # noqa: E402
from probe import RESULTS_DIR, LogReg, metrics  # noqa: E402
from splits import data_signature, split_rows, usable  # noqa: E402

PRECISIONS = ["FP32_GPU", "FP16_GPU", "BF16_GPU", "FP32_CPU", "INT8_CPU"]
BATCH, WARMUP, REPEATS = 32, 3, 3
DTYPE = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16, "INT8": torch.float32}


def subset_rows(df, dataset, n):
    test = split_rows(df, dataset, 42)["test"]
    if len(test) > n:
        test = np.sort(np.random.RandomState(42).choice(test, n, replace=False))
    return test


def load_images(df, rows, transform):
    return torch.stack([transform(Image.open(os.path.join(BASE_DIR, df.path.values[r])).convert("RGB")) for r in rows])


def probe_record(model, dataset, probe_dir):
    return json.load(open(os.path.join(probe_dir, model, dataset, "seed42.json")))


def is_current(json_path, probe):
    """A precision result counts as done only if it was made with the current seed-42 probe (same C and same input
    data, i.e. splits.data_signature incl. the tissue filter); otherwise it is recomputed."""
    if not os.path.exists(json_path):
        return False
    p = json.load(open(json_path)).get("probe", {})
    return p.get("C") == probe["C"] and p.get("data_signature") == probe.get("data_signature")


def refit_probe(model, dataset, X, df, probe_dir, device):
    r = probe_record(model, dataset, probe_dir)
    if r.get("data_signature") != data_signature(dataset, df):
        raise RuntimeError(f"{model}/{dataset}: probe seed42.json was made from other data (rerun probe.py)")
    parts = split_rows(df, dataset, 42)
    fit = np.concatenate([parts["train"], parts["val"]])
    return LogReg(r["C"], device).fit(X[fit], df.label.values[fit], r["n_classes"]), r


def build(extractor, precision):
    """(module, device, input dtype) for one precision; the FP32 extractor itself is never modified."""
    kind, dev = precision.split("_")
    if kind == "INT8":
        m = copy.deepcopy(extractor).cpu().float()
        return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval(), torch.device("cpu"), torch.float32
    device = torch.device("cuda") if dev == "GPU" else torch.device("cpu")
    m = copy.deepcopy(extractor).to(device)
    if kind != "FP32":
        m = m.to(DTYPE[kind])
    return m.eval(), device, DTYPE[kind]


@torch.no_grad()
def embed_and_time(module, device, dtype, images):
    feats = []
    for i in range(0, len(images), BATCH):
        feats.append(module(images[i:i + BATCH].to(device, dtype))["official"].float().cpu())
    full = [images[i:i + BATCH].to(device, dtype) for i in range(0, len(images) - BATCH + 1, BATCH)]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for i in range(WARMUP):
        module(full[i % len(full)])
    sync()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for x in full:
            module(x)
        sync()
        times.append(time.perf_counter() - t0)
    n = BATCH * len(full)
    ms = [1000 * t / n for t in times]
    return torch.cat(feats).numpy(), {
        "ms_per_image_mean": float(np.mean(ms)), "ms_per_image_sd": float(np.std(ms, ddof=1)),
        "images_per_s_mean": float(np.mean([n / t for t in times])), "images_per_s_sd": float(np.std([n / t for t in times], ddof=1)),
        "n_images_timed": n, "batch_size": BATCH, "warmup_batches": WARMUP, "repeats": REPEATS, "times_s": times,
        "cpu_threads": torch.get_num_threads(), "device": str(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["lung", "breakhis", "nct", "hubmap", "bach"])
    ap.add_argument("--precisions", nargs="+", default=PRECISIONS, choices=PRECISIONS)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--features", default=FEATURES_DIR)
    ap.add_argument("--probe-dir", default=os.path.join(RESULTS_DIR, "probe"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "precision"))
    args = ap.parse_args()

    from extract import load_features
    from models import MODELS
    set_strict_fp32()
    torch.backends.quantized.engine = "qnnpack" if platform.machine().startswith(("arm", "aarch64")) else "fbgemm"
    env = dict(environment_info(), quantized_engine=torch.backends.quantized.engine, cpu=platform.processor() or platform.machine())
    for model in args.models:
        extractor, transform, meta = MODELS[model]()
        for dataset in args.datasets:
            out_dir = os.path.join(args.out, model, dataset)
            probe = probe_record(model, dataset, args.probe_dir)
            current = {p: is_current(os.path.join(out_dir, f"{p}.json"), probe) for p in args.precisions}
            if not current.get("FP32_GPU", True):  # the reference changed: every precision is redone
                current = {p: False for p in current}
            if all(current.values()):
                print(f"[skip] {model}/{dataset}", flush=True)
                continue
            X, df = load_features(model, dataset, root=args.features)
            keep = usable(dataset, df)  # the same tissue filter as probe.py (PANDA, BACH, HuBMAP)
            X, df = X[keep], df[keep]
            lr, probe_meta = refit_probe(model, dataset, X, df, args.probe_dir, torch.device("cuda"))
            rows = subset_rows(df, dataset, args.n_images)
            images = load_images(df, rows, transform)
            y = df.label.values[rows]
            ref_path = os.path.join(out_dir, "FP32_GPU.npz")
            for precision in args.precisions:
                json_path = os.path.join(out_dir, f"{precision}.json")
                if current[precision]:
                    continue
                if precision != "FP32_GPU" and not os.path.exists(ref_path):
                    raise RuntimeError("FP32_GPU must run first (it is the reference)")
                module, device, dtype = build(extractor, precision)
                feats, timing = embed_and_time(module, device, dtype, images)
                del module
                torch.cuda.empty_cache()
                probs = lr.predict_proba(feats)
                out = {"model": model, "dataset": dataset, "precision": precision, "n_images": int(len(rows)),
                       "metrics": metrics(y, probs, probe_meta["n_classes"]), "timing": timing,
                       "probe": {"seed": 42, "C": probe_meta["C"], "data_signature": probe_meta["data_signature"]},
                       "filter": probe_meta.get("filter"), "model_meta": meta, "environment": env}
                if precision != "FP32_GPU":
                    ref = np.load(ref_path)
                    cos = (feats * ref["feats"]).sum(1) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(ref["feats"], axis=1))
                    out["vs_fp32_gpu"] = {"prediction_agreement": float((probs.argmax(1) == ref["probs"].argmax(1)).mean()),
                                          "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min())}
                buf = io.BytesIO()
                np.savez_compressed(buf, rows=rows, targets=y, probs=probs.astype(np.float32), feats=feats.astype(np.float32))
                atomic_write_bytes(os.path.join(out_dir, f"{precision}.npz"), buf.getvalue())
                atomic_write_json(json_path, out)
                agree = out.get("vs_fp32_gpu", {}).get("prediction_agreement", 1.0)
                print(f"{model:10s} {dataset:9s} {precision:9s} acc {out['metrics']['acc']:.4f} agree {agree:.4f} "
                      f"{timing['ms_per_image_mean']:.2f}±{timing['ms_per_image_sd']:.2f} ms/img", flush=True)
        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
