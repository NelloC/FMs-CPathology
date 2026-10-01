"""Slide-level ISUP grading on PANDA with gated-attention MIL (Ilse et al., 2018) over cached tile embeddings.

Same slides and seeds as the patch-level probe: the slide-disjoint train/val/test parts come from
splits.split_rows (tile rows grouped by slide). Slide label = the ISUP grade of the slide's folder
(every slide's tiles are in one folder).

Model: Linear(d, 256) + ReLU + dropout 0.25 -> gated attention (256 -> 128) -> Linear(256, 6).
Training: AdamW lr 2e-4, weight decay 1e-4, cross-entropy, up to 50 epochs, 32 slides per batch (bags padded
and masked); stops after 10 epochs without a better validation QWK; the epoch with the best validation QWK
is kept; the test set is evaluated once with it.
Only tiles with >= 10% tissue are used (splits.usable; the rest are black/white padding from the tiling).
A unit = model x seed; a unit whose .json was made from the same input (splits.data_signature) and the same
hyperparameters is skipped.

Output: results/abmil/<model>/panda/seed<k>.{json,npz}; the .npz has the fields stats.py reads
(rows = slide numbers, groups = slide ids, targets, probs), so
  python pipeline/stats.py --probe-dir results/abmil --out results/stats_abmil --datasets panda
gives the same CIs, paired tests and ranks at slide level.
"""
import argparse
import copy
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES_DIR, atomic_write_bytes, atomic_write_json, environment_info, set_strict_fp32  # noqa: E402
from probe import RESULTS_DIR, metrics  # noqa: E402
from splits import check_disjoint, data_signature, filter_config, split_rows, usable  # noqa: E402

N_CLASSES = 6
EPOCHS, PATIENCE, BATCH, LR, WD = 50, 10, 32, 2e-4, 1e-4
HPARAMS = dict(epochs=EPOCHS, patience=PATIENCE, batch_slides=BATCH, lr=LR, weight_decay=WD, hidden=256, attn=128, dropout=0.25)


class GatedABMIL(nn.Module):
    def __init__(self, d, hidden=256, attn=128, n_classes=N_CLASSES, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.v = nn.Sequential(nn.Linear(hidden, attn), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(hidden, attn), nn.Sigmoid())
        self.w = nn.Linear(attn, 1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, mask):
        """x [B, T, d], mask [B, T] (True = real tile) -> logits [B, C], attention [B, T]."""
        h = self.embed(x)
        a = self.w(self.v(h) * self.u(h)).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        return self.head((a.unsqueeze(-1) * h).sum(1)), a


def make_bags(X, df):
    """Per slide: tile rows (index order), label; returns slide ids, list of row arrays, labels, slide of each row."""
    slides, slide_of_row = np.unique(df.group.values, return_inverse=True)
    order = np.argsort(slide_of_row, kind="stable")
    bounds = np.searchsorted(slide_of_row[order], np.arange(len(slides) + 1))
    rows = [order[bounds[i]:bounds[i + 1]] for i in range(len(slides))]
    labels = np.array([df.label.values[r[0]] for r in rows])
    for r, lab in zip(rows, labels):
        if (df.label.values[r] != lab).any():
            raise RuntimeError("tiles of one slide carry different labels")
    return slides, rows, labels, slide_of_row


def batches(X, rows, slide_ids, device, shuffle, gen=None):
    idx = np.array(slide_ids)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen).numpy()]
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        t = max(len(rows[s]) for s in b)
        x = torch.zeros(len(b), t, X.shape[1])
        mask = torch.zeros(len(b), t, dtype=torch.bool)
        for j, s in enumerate(b):
            x[j, :len(rows[s])] = torch.from_numpy(X[rows[s]])
            mask[j, :len(rows[s])] = True
        yield b, x.to(device, non_blocking=True), mask.to(device)


@torch.no_grad()
def predict(model, X, rows, slide_ids, device):
    model.eval()
    probs = {}
    for b, x, mask in batches(X, rows, slide_ids, device, shuffle=False):
        p = torch.softmax(model(x, mask)[0], 1).cpu().numpy()
        probs.update(zip(b, p))
    return np.stack([probs[s] for s in slide_ids])


def is_done(json_path, signature):
    if not os.path.exists(json_path):
        return False
    r = json.load(open(json_path))
    return r.get("data_signature") == signature and r.get("hyperparameters") == HPARAMS


def run_unit(model_name, seed, X, df, device, out_dir):
    json_path = os.path.join(out_dir, f"seed{seed}.json")
    signature = data_signature("panda", df)
    if is_done(json_path, signature):
        return "skip"
    t0 = time.time()
    parts = split_rows(df, "panda", seed)
    check_disjoint(df, parts)
    slides, rows, labels, slide_of_row = make_bags(X, df)
    sets = {k: np.unique(slide_of_row[v]) for k, v in parts.items()}

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = GatedABMIL(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    history, best = [], (-np.inf, None, 0)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for b, x, mask in batches(X, rows, sets["train"], device, shuffle=True, gen=gen):
            loss = nn.functional.cross_entropy(model(x, mask)[0], torch.as_tensor(labels[b], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        m = metrics(labels[sets["val"]], predict(model, X, rows, sets["val"], device), N_CLASSES)
        history.append(dict(epoch=epoch + 1, train_loss=total / len(sets["train"]), val=m))
        if m["qwk"] > best[0]:
            best = (m["qwk"], copy.deepcopy(model.state_dict()), epoch + 1)
        elif epoch + 1 - best[2] >= PATIENCE:
            break
    model.load_state_dict(best[1])
    test = sets["test"]
    probs = predict(model, X, rows, test, device)

    buf = io.BytesIO()
    np.savez_compressed(buf, rows=test, groups=slides[test].astype(str), paths=slides[test].astype(str),
                        targets=labels[test], probs=probs.astype(np.float32))
    atomic_write_bytes(os.path.join(out_dir, f"seed{seed}.npz"), buf.getvalue())
    atomic_write_json(json_path, {
        "model": model_name, "dataset": "panda", "seed": seed, "level": "slide", "filter": filter_config("panda"),
        "data_signature": signature, "method": "gated ABMIL",
        "n_classes": N_CLASSES, "best_epoch": best[2], "history": history, "test": metrics(labels[test], probs, N_CLASSES),
        "hyperparameters": HPARAMS,
        "counts": {k: {"slides": int(len(v)), "tiles": int(sum(len(rows[s]) for s in v))} for k, v in sets.items()},
        "label_source": "ISUP folder of the slide's tiles", "seconds": round(time.time() - t0, 1),
    })
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--features", default=FEATURES_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "abmil"))
    args = ap.parse_args()
    from extract import load_features
    set_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = []
    for model_name in args.models:
        out_dir = os.path.join(args.out, model_name, "panda")
        from data_index import load_index
        idx = load_index("panda")
        signature = data_signature("panda", idx[usable("panda", idx)])
        if all(is_done(os.path.join(out_dir, f"seed{s}.json"), signature) for s in args.seeds):
            print(f"[skip] {model_name}", flush=True)
            continue
        try:
            X, df = load_features(model_name, "panda", root=args.features)
        except RuntimeError as e:
            print(f"[wait] {model_name}: {e}", flush=True)
            continue
        keep = usable("panda", df)
        X, df = X[keep], df[keep]
        os.makedirs(out_dir, exist_ok=True)
        atomic_write_json(os.path.join(out_dir, "meta.json"), {"environment": environment_info()})
        for seed in args.seeds:
            try:
                status = run_unit(model_name, seed, X, df, device, out_dir)
            except Exception as e:  # one failing unit must not stop the others; a rerun retries it
                failures.append(f"{model_name}/seed{seed}: {e!r}")
                print(f"[FAILED] {failures[-1]}", flush=True)
                torch.cuda.empty_cache()
                continue
            if status == "done":
                r = json.load(open(os.path.join(out_dir, f"seed{seed}.json")))
                print(f"{model_name:10s} seed {seed} | best epoch {r['best_epoch']} | test QWK {r['test']['qwk']:.4f} "
                      f"acc {r['test']['acc']:.4f} | {r['seconds']}s", flush=True)
    if failures:
        print(f"{len(failures)} unit(s) failed:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
