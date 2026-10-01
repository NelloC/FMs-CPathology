"""Per-image tissue statistics, used to exclude non-tissue tiles (e.g. PANDA tiles made of black padding).

A pixel is tissue when it is coloured and not dark: max(R,G,B) - min(R,G,B) > 15 and max(R,G,B) > 30.
This rejects black padding (0,0,0), white/grey background and glass. Also recorded: fraction of pure
black pixels and of near-white pixels (all channels > 220), and an MD5 of the file (exact duplicates).
Statistics are computed on the image as stored (PANDA tiles 256 px; BRACS RoIs downsampled to <= 1024 px).

Output: indexes/tissue_<dataset>.csv, rows aligned with indexes/<dataset>.csv.
Usage: python pipeline/tissue.py [dataset ...]
"""
import hashlib
import io
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, INDEX_DIR, atomic_write_bytes  # noqa: E402
from data_index import load_index  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def tile_stats(rel):
    path = os.path.join(BASE_DIR, rel)
    raw = open(path, "rb").read()
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if max(im.size) > 1024:
            im.thumbnail((1024, 1024))
        a = np.asarray(im).astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    return ((mx - mn > 15) & (mx > 30)).mean(), (mx == 0).mean(), (mn > 220).mean(), hashlib.md5(raw).hexdigest()


def tissue_table(dataset, workers=10):
    path = os.path.join(INDEX_DIR, f"tissue_{dataset}.csv")
    df = load_index(dataset)
    if os.path.exists(path):
        t = pd.read_csv(path)
        if t.path.tolist() == df.path.tolist():
            return t
    with Pool(workers) as p:
        stats = p.map(tile_stats, df.path.tolist(), chunksize=256)
    t = pd.DataFrame(stats, columns=["tissue_frac", "black_frac", "white_frac", "md5"])
    t.insert(0, "path", df.path.values)
    buf = io.StringIO()
    t.to_csv(buf, index=False, float_format="%.4f")
    atomic_write_bytes(path, buf.getvalue().encode())
    return t


if __name__ == "__main__":
    for name in sys.argv[1:] or ["lung", "breakhis", "nct", "hubmap", "bach", "sicap", "bracs", "panda"]:
        t = tissue_table(name)
        q = t.tissue_frac.quantile([0, .01, .05, .5]).round(3).tolist()
        print(f"{name:9s} n={len(t):7d} tissue_frac q0/1/5/50%={q} | <5%: {(t.tissue_frac < .05).sum()} "
              f"<10%: {(t.tissue_frac < .10).sum()} <25%: {(t.tissue_frac < .25).sum()} | "
              f"duplicate files: {t.md5.duplicated().sum()}", flush=True)
