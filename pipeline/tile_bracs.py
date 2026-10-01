"""Fixed-magnification tiling of the BRACS RoIs (sensitivity analysis to the whole-RoI resize of extract.py).

Every RoI of indexes/bracs.csv is resampled from its native resolution (NATIVE_MPP) to TARGET_MPP, then cut into
non-overlapping TILE x TILE tiles:
  * resampling: PIL LANCZOS (anti-aliased) to round(w * NATIVE_MPP / TARGET_MPP) x round(h * ...);
  * grid: floor(W / TILE) x floor(H / TILE) tiles, centred on the RoI (the leftover margin, < TILE px per axis,
    is split evenly between both sides); partial edge tiles are dropped;
  * RoIs with a side < TILE after resampling: that side is padded with white (255) to TILE, centred, so the
    tissue keeps the fixed resolution and none of it is cropped away (a crop cannot enlarge a side < TILE);
    along a side >= TILE the grid is built as usual. Padding is background by the tissue definition below;
  * tissue filter: tissue fraction per tile with pipeline/tissue.py's pixel definition
    (max(RGB) - min(RGB) > 15 and max(RGB) > 30); tiles with fraction >= TISSUE_MIN are kept. If no tile of an
    RoI passes, its single most-tissue tile is kept (ties: first in raster order) and the RoI is flagged.

Output (OUT_DIR, default <BASE_DIR>/dataset_derived/bracs_tiles_0p5um):
  tiles/<roi_row:04d>_<roi stem>/y<y>_x<x>.png   x, y: top-left corner in pixels of the resampled RoI
                                                  (before padding; negative when padded)
  rois/<roi_row:04d>.csv   per-RoI tile list, written atomically after its PNGs (resume marker)
  tiles.csv                tile index: path (relative to BASE_DIR), roi_row (row of indexes/bracs.csv), roi_path,
                           x, y, tissue_frac, label, class_name, partition, fallback; sorted by roi_row, y, x
  meta.json                configuration, resolution evidence, counts; a rerun with another configuration aborts
  DONE

Usage: python pipeline/tile_bracs.py [--workers 6] [--out DIR]
"""
import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, atomic_write_bytes, atomic_write_json, sha256_strings  # noqa: E402
from data_index import index_hash, load_index  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

NATIVE_MPP = 0.25
TARGET_MPP = 0.5
TILE = 224
TISSUE_MIN = 0.10
PAD_VALUE = 255
OUT_DIR = os.path.join(BASE_DIR, "dataset_derived", "bracs_tiles_0p5um")

RESOLUTION_EVIDENCE = {
    "native_mpp": NATIVE_MPP,
    "sources": [
        "Brancati et al., 'BRACS: A Dataset for BReAst Carcinoma Subtyping in H&E Histology Images', "
        "arXiv:2111.04740 (Database, 2022, baac093), section 'BRACS dataset characteristics': 'All slides were "
        "scanned with an Aperio AT2 scanner at 0.25 um/pixel using a magnification factor of 40x.'",
        "Same paper, Table 4 (general information on BRACS): RoI 'Size (Magnification)' = '4537 (40x)', "
        "WSI = '547 (40x)'; RoIs extracted with QuPath from the WSIs.",
        "Dataset website https://www.bracs.icar.cnr.it/details/: 'The resolution of each RoI is 40x'.",
    ],
    "file_metadata": "RoI PNGs carry no resolution metadata (no pHYs/dpi/text chunks; checked on every 20th file "
                     "of the index, 227 files).",
}


def tissue_mask(a):
    """tissue.py's pixel definition on a uint8 RGB array."""
    a = a.astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    return (mx - mn > 15) & (mx > 30)


def resample(img, native_mpp=NATIVE_MPP, target_mpp=TARGET_MPP):
    s = native_mpp / target_mpp
    size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
    return img if size == img.size else img.resize(size, Image.Resampling.LANCZOS)


def _axis(n, tile):
    """Tile origins along one axis: centred grid if n >= tile, else a single origin -(tile - n) // 2 (padding)."""
    if n < tile:
        return [-((tile - n) // 2)]
    k = n // tile
    off = (n - k * tile) // 2
    return [off + i * tile for i in range(k)]


def cut_tiles(img, tile=TILE, tissue_min=TISSUE_MIN):
    """img: RGB PIL image already at the target resolution.
    Returns (list of (x, y, tissue_frac, PIL tile)), fallback flag). Always at least one tile."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    xs, ys = _axis(w, tile), _axis(h, tile)
    if w < tile or h < tile:  # pad with white, centred
        ph, pw = max(h, tile), max(w, tile)
        oy, ox = max(0, (tile - h) // 2), max(0, (tile - w) // 2)
        p = np.full((ph, pw, 3), PAD_VALUE, np.uint8)
        p[oy:oy + h, ox:ox + w] = a
        a, dx, dy = p, ox, oy
    else:
        dx = dy = 0
    mask = tissue_mask(a)
    cand = []
    for y in ys:
        for x in xs:
            fy, fx = y + dy, x + dx
            cand.append((x, y, float(mask[fy:fy + tile, fx:fx + tile].mean())))
    keep = [c for c in cand if c[2] >= tissue_min]
    fallback = not keep
    if fallback:
        keep = [max(cand, key=lambda c: c[2])]  # max() returns the first of ties (raster order)
    out = [(x, y, t, Image.fromarray(np.ascontiguousarray(a[y + dy:y + dy + tile, x + dx:x + dx + tile])))
           for x, y, t in keep]
    return out, fallback


def tile_dir_name(roi_row, roi_path):
    return f"{roi_row:04d}_{os.path.splitext(os.path.basename(roi_path))[0]}"


def process_roi(job):
    """job: (roi_row, roi_path, out_dir, config). Writes the tiles, then rois/<row>.csv. Returns the row."""
    roi_row, roi_path, out_dir, cfg = job
    marker = os.path.join(out_dir, "rois", f"{roi_row:04d}.csv")
    if os.path.exists(marker):
        return roi_row
    with Image.open(os.path.join(BASE_DIR, roi_path)) as im:
        img = im.convert("RGB")
    orig = img.size
    img = resample(img, cfg["native_mpp"], cfg["target_mpp"])
    tiles, fallback = cut_tiles(img, cfg["tile"], cfg["tissue_min"])
    del img
    tdir = os.path.join(out_dir, "tiles", tile_dir_name(roi_row, roi_path))
    rel_out = os.path.relpath(tdir, BASE_DIR)
    rows = []
    for x, y, t, im in tiles:
        name = f"y{y}_x{x}.png"
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
        atomic_write_bytes(os.path.join(tdir, name), buf.getvalue())
        rows.append(dict(path=os.path.join(rel_out, name), roi_row=roi_row, roi_path=roi_path, x=x, y=y,
                         tissue_frac=round(t, 6), fallback=int(fallback), roi_w=orig[0], roi_h=orig[1]))
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    atomic_write_bytes(marker, buf.getvalue().encode())
    return roi_row


def tiling_config(df):
    return {"native_mpp": NATIVE_MPP, "target_mpp": TARGET_MPP, "tile": TILE, "tissue_min": TISSUE_MIN,
            "resample": "PIL LANCZOS, size = round(side * native_mpp / target_mpp)",
            "grid": "non-overlapping, centred, partial edge tiles dropped",
            "small_side": f"side < tile padded with white ({PAD_VALUE}) to tile, centred",
            "tissue": "tissue.py pixel definition: max-min > 15 and max > 30 (padding counts as non-tissue)",
            "fallback": "no tile >= tissue_min -> keep the single most-tissue tile",
            "png": "compress_level=1",
            "index": "bracs", "index_hash": index_hash(df), "n_rois": len(df)}


def run(out_dir=OUT_DIR, workers=6, df=None):
    """Tiles every RoI of df (default: the BRACS index); returns the tile table. Resumable."""
    df = load_index("bracs") if df is None else df
    cfg = tiling_config(df)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))["config"]
        if old != cfg:
            diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
            raise RuntimeError(f"{out_dir}: existing tiles were made with a different configuration: {diff}")
    else:
        atomic_write_json(meta_path, {"config": cfg, "resolution_evidence": RESOLUTION_EVIDENCE,
                                      "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print(f"[skip] tiling complete: {out_dir}", flush=True)
        return pd.read_csv(os.path.join(out_dir, "tiles.csv"), keep_default_na=False)

    jobs = [(i, p, out_dir, cfg) for i, p in enumerate(df.path)
            if not os.path.exists(os.path.join(out_dir, "rois", f"{i:04d}.csv"))]
    # biggest RoIs first: better load balance, and memory peaks happen while the pool is still full anyway
    area = {i: np.prod(Image.open(os.path.join(BASE_DIR, p)).size) for i, p, _, _ in jobs}
    jobs.sort(key=lambda j: -area[j[0]])
    print(f"[run] tiling {len(df)} RoIs, {len(jobs)} pending, {workers} workers -> {out_dir}", flush=True)
    t0 = time.time()
    if jobs:
        with Pool(workers, maxtasksperchild=50) as pool:
            for n, _ in enumerate(pool.imap_unordered(process_roi, jobs, chunksize=1), 1):
                if n % 250 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} RoIs | {time.time() - t0:.0f}s", flush=True)

    parts = [pd.read_csv(os.path.join(out_dir, "rois", f"{i:04d}.csv")) for i in range(len(df))]
    t = pd.concat(parts, ignore_index=True)
    t = t.merge(df[["label", "class_name", "partition"]], left_on="roi_row", right_index=True, how="left")
    t = t.sort_values(["roi_row", "y", "x"], kind="stable").reset_index(drop=True)
    t = t[["path", "roi_row", "roi_path", "x", "y", "tissue_frac", "label", "class_name", "partition", "fallback",
           "roi_w", "roi_h"]]
    if (t.roi_path.values != df.path.values[t.roi_row.values]).any() or t.roi_row.nunique() != len(df):
        raise RuntimeError("tile table does not match the BRACS index")
    missing = [p for p in t.path if not os.path.isfile(os.path.join(BASE_DIR, p))]
    if missing:
        raise RuntimeError(f"{len(missing)} tile files missing, e.g. {missing[:3]}")
    buf = io.StringIO()
    t.to_csv(buf, index=False)
    atomic_write_bytes(os.path.join(out_dir, "tiles.csv"), buf.getvalue().encode())

    per_roi = t.groupby("roi_row").size()
    roi = df.assign(n_tiles=per_roi.reindex(range(len(df))).values)
    small = [min(round(w * NATIVE_MPP / TARGET_MPP), round(h * NATIVE_MPP / TARGET_MPP)) < TILE
             for w, h in t.groupby("roi_row")[["roi_w", "roi_h"]].first().values]
    counts = {
        "tiles": int(len(t)), "rois": int(len(df)),
        "tiles_per_roi": {k: float(v) for k, v in per_roi.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
                          .items()},
        "rois_padded_side_lt_tile": int(sum(small)),
        "rois_fallback_single_tile": int(t.groupby("roi_row").fallback.first().sum()),
        "per_class": {c: {"rois": int(len(g)), "tiles": int(g.n_tiles.sum()), "median_tiles": float(g.n_tiles.median()),
                          "mean_tiles": round(float(g.n_tiles.mean()), 2)} for c, g in roi.groupby("class_name")},
        "per_partition": {p: int(g.n_tiles.sum()) for p, g in roi.groupby("partition")},
        "tile_index_hash": sha256_strings(t.path.tolist()),
    }
    meta = json.load(open(meta_path))
    meta.update(counts=counts, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_write_json(meta_path, meta)
    atomic_write_bytes(os.path.join(out_dir, "DONE"), time.strftime("%Y-%m-%d %H:%M:%S\n").encode())
    print(json.dumps(counts, indent=1), flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(args.out, args.workers)


if __name__ == "__main__":
    main()
