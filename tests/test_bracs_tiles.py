"""BRACS fixed-magnification arm: tiling geometry (pipeline/tile_bracs.py), pooling and load_features compatibility
(pipeline/extract_tiles.py), resume; plus a GPU smoke test of the real extraction on a few tiles."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from conftest import CODE_DIR, PIPELINE_DIR
from common import BASE_DIR  # noqa: E402

import extract  # noqa: E402
import extract_tiles as et  # noqa: E402
import tile_bracs as tb  # noqa: E402
from common import atomic_write_bytes, atomic_write_json, sha256_strings  # noqa: E402
from data_index import index_hash, load_index  # noqa: E402

TISSUE = (180, 80, 160)  # H&E-like pink: tissue by tissue.py's definition
WHITE = (255, 255, 255)


def img(w, h, color=TISSUE):
    return Image.new("RGB", (w, h), color)


# ---------------------------------------------------------------- tiling geometry

def test_tissue_definition_matches_tissue_py(tmp_path):
    a = np.array([[TISSUE, WHITE, (0, 0, 0), (200, 200, 190), (40, 20, 30), (30, 10, 20)]], np.uint8)
    assert tb.tissue_mask(a).tolist() == [[True, False, False, False, True, False]]
    import tissue
    rng = np.random.default_rng(0)
    r = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    p = tmp_path / "t.png"
    Image.fromarray(r).save(p)
    assert tissue.tile_stats(os.path.relpath(p, BASE_DIR))[0] == tb.tissue_mask(r).mean()


def test_resample_halves_native_40x():
    assert tb.NATIVE_MPP / tb.TARGET_MPP == 0.5
    assert tb.resample(img(1000, 601)).size == (500, 300)  # round(300.5) -> 300 (banker's) is fine either way
    assert tb.resample(img(1000, 600), 0.25, 0.25).size == (1000, 600)
    assert tb.resample(img(900, 900), 0.25, 1.0).size == (225, 225)


def test_grid_count_and_centring():
    tiles, fb = tb.cut_tiles(img(500, 300))
    assert not fb
    assert [(x, y) for x, y, _, _ in tiles] == [(26, 38), (250, 38)]  # 2 x 1, margins split evenly
    assert all(t.size == (224, 224) and f == 1.0 for _, _, f, t in tiles)
    tiles, _ = tb.cut_tiles(img(224 * 3, 224 * 2))
    assert len(tiles) == 6 and tiles[0][:2] == (0, 0)
    tiles, _ = tb.cut_tiles(img(447, 447))  # partial edge tiles dropped
    assert len(tiles) == 1


def test_tile_content_is_the_crop():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (300, 500, 3), dtype=np.uint8)
    for x, y, _, t in tb.cut_tiles(Image.fromarray(a), tissue_min=0)[0]:
        assert np.array_equal(np.asarray(t), a[y:y + 224, x:x + 224])


def test_small_roi_padded_white():
    tiles, fb = tb.cut_tiles(img(150, 300))  # width < tile: padded, one row of tiles along the height
    assert len(tiles) == 1 and not fb
    x, y, f, t = tiles[0]
    assert (x, y) == (-37, 38)
    a = np.asarray(t)
    assert a.shape == (224, 224, 3)
    assert (a[:, :37] == 255).all() and (a[:, 37 + 150:] == 255).all() and (a[:, 37:187] == TISSUE).all()
    assert f == pytest.approx(150 / 224)
    tiles, _ = tb.cut_tiles(img(100, 80))  # both sides small
    assert len(tiles) == 1 and tiles[0][2] == pytest.approx(100 * 80 / 224 ** 2)
    tiles, _ = tb.cut_tiles(img(150, 600))  # one small side, grid along the other
    assert len(tiles) == 2


def test_tissue_filter_and_fallback():
    im = img(672, 224, WHITE)
    im.paste(img(224, 224), (224, 0))  # only the middle tile is tissue
    im.paste(img(20, 224), (0, 0))     # left tile: 20/224 = 8.9% tissue -> dropped
    tiles, fb = tb.cut_tiles(im)
    assert not fb and [(x, f) for x, _, f, _ in tiles] == [(224, 1.0)]
    im = img(672, 224, WHITE)
    im.paste(img(10, 224), (224 * 2, 0))  # all below 10%: keep the most-tissue tile only
    tiles, fb = tb.cut_tiles(im)
    assert fb and len(tiles) == 1 and tiles[0][0] == 448
    tiles, fb = tb.cut_tiles(img(672, 224, WHITE))  # blank: first tile in raster order
    assert fb and tiles[0][:2] == (0, 0)


# ---------------------------------------------------------------- tiling run: files, table, resume

def _synthetic_rois(root):
    """Four PNG 'RoIs' at native 0.25 um/px under root; returns an index-like DataFrame (paths rel. to BASE_DIR)."""
    specs = [(1000, 600, TISSUE), (300, 700, TISSUE), (900, 460, WHITE), (2000, 2000, TISSUE)]
    rows = []
    for i, (w, h, c) in enumerate(specs):
        p = os.path.join(root, "rois", f"BRACS_{i}_X_{i}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        im = img(w, h, c)
        if i == 3:
            im.paste(img(1000, 2000, WHITE), (0, 0))  # left half blank
        im.save(p)
        rows.append(dict(path=os.path.relpath(p, BASE_DIR), label=i, class_name=f"{i}_X", partition="train"))
    return pd.DataFrame(rows)


def test_tiling_run_resume_and_config_guard(tmp_path):
    df = _synthetic_rois(str(tmp_path))
    out = str(tmp_path / "tiles")
    t = tb.run(out, workers=2, df=df)
    # RoI 0: 500x300 -> 2; RoI 1: 150x350 -> 1 (padded); RoI 2: blank 450x230 -> 2 candidates, fallback 1;
    # RoI 3: 1000x1000 -> 4x4 grid, left half blank -> 8 tiles
    assert t.groupby("roi_row").size().tolist() == [2, 1, 1, 8]
    assert t.groupby("roi_row").fallback.first().tolist() == [0, 0, 1, 0]
    assert (t.roi_path.values == df.path.values[t.roi_row]).all()
    assert all(Image.open(os.path.join(BASE_DIR, p)).size == (224, 224) for p in t.path)
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["counts"]["tiles"] == 12 and meta["counts"]["rois_fallback_single_tile"] == 1
    assert meta["counts"]["rois_padded_side_lt_tile"] == 1
    assert meta["resolution_evidence"]["native_mpp"] == 0.25
    before = {p: open(os.path.join(BASE_DIR, p), "rb").read() for p in t.path}
    # simulated crash: one RoI's marker and a tile lost, DONE not written yet
    os.remove(os.path.join(out, "DONE"))
    os.remove(os.path.join(out, "rois", "0003.csv"))
    os.remove(os.path.join(BASE_DIR, t.path.iloc[-1]))
    t2 = tb.run(out, workers=2, df=df)
    pd.testing.assert_frame_equal(t, t2)
    assert before == {p: open(os.path.join(BASE_DIR, p), "rb").read() for p in t2.path}
    assert tb.run(out, workers=2, df=df).equals(t)  # DONE: skipped
    with pytest.raises(RuntimeError, match="different configuration"):
        tb.run(out, workers=2, df=df.iloc[:3])


# ---------------------------------------------------------------- pooling and load_features compatibility

def _fake_tree(tmp_path, n_per_roi, shard_size=1000, keys=("official", "cls"), dims=(6, 4)):
    """Tile index covering the real BRACS index + fake per-tile features in extract.py's layout."""
    df = load_index("bracs")
    roi = np.repeat(np.arange(len(df)), n_per_roi)
    tiles = pd.DataFrame({"path": [f"fake/{r}_{i}.png" for i, r in enumerate(roi)], "roi_row": roi,
                          "roi_path": df.path.values[roi]})
    tdir = str(tmp_path / "tiles")
    os.makedirs(tdir)
    tiles.to_csv(os.path.join(tdir, "tiles.csv"), index=False)
    tmeta = {"config": {"index_hash": index_hash(df), "tile": 224},
             "counts": {"tile_index_hash": sha256_strings(tiles.path.tolist()), "tiles": len(tiles),
                        "tiles_per_roi": {}}, "resolution_evidence": {"native_mpp": 0.25}}
    atomic_write_json(os.path.join(tdir, "meta.json"), tmeta)
    open(os.path.join(tdir, "DONE"), "w").write("x\n")

    from safetensors.torch import save
    rng = np.random.default_rng(1)
    feats = {k: rng.standard_normal((len(tiles), d)).astype(np.float32) for k, d in zip(keys, dims)}
    src = os.path.join(str(tmp_path / "ft"), "fake", et.TILE_DATASET)
    n_shards = (len(tiles) + shard_size - 1) // shard_size
    for k in range(n_shards):
        lo, hi = k * shard_size, min(len(tiles), (k + 1) * shard_size)
        t = {key: torch.from_numpy(v[lo:hi].copy()) for key, v in feats.items()}
        t["row"] = torch.arange(lo, hi, dtype=torch.int64)
        atomic_write_bytes(extract.shard_path(src, k), save(t))
    config = {"model": "fake", "dataset": et.TILE_DATASET, "model_meta": {"tiles": et.tiling_signature(tmeta)},
              "transform": "T", "index_hash": index_hash(tiles), "n_index_rows": len(tiles), "subset": 0,
              "rows_hash": index_hash(tiles), "n_rows": len(tiles), "shard_size": shard_size,
              "precision": "fp32", "libs": {}}
    atomic_write_json(os.path.join(src, "meta.json"), {"config": config, "environment": {}})
    open(os.path.join(src, "DONE"), "w").write("x\n")
    return df, tiles, tdir, feats


def test_pooling_mean_and_load_features(tmp_path):
    n_per_roi = (np.arange(len(load_index("bracs"))) % 3) + 1  # 1..3 tiles per RoI
    df, tiles, tdir, feats = _fake_tree(tmp_path, n_per_roi)
    pooled = str(tmp_path / "pooled")
    et.pool("fake", tdir, str(tmp_path / "ft"), pooled)
    for key, v in feats.items():
        X, d = extract.load_features("fake", "bracs", key=key, root=pooled)
        assert X.dtype == np.float32 and X.shape == (len(df), v.shape[1])
        pd.testing.assert_frame_equal(d, df)
        expect = pd.DataFrame(v.astype(np.float64)).groupby(tiles.roi_row.values).mean().values
        np.testing.assert_allclose(X, expect, rtol=1e-6, atol=1e-6)
    # a single-tile RoI's pooled vector is exactly its tile's vector
    i = int(np.flatnonzero(n_per_roi == 1)[0])
    X, _ = extract.load_features("fake", "bracs", root=pooled)
    assert np.array_equal(X[i], feats["official"][tiles.roi_row.values == i][0])
    meta = json.load(open(os.path.join(pooled, "fake", "bracs", "meta.json")))
    assert meta["config"]["index_hash"] == index_hash(df) and meta["config"]["n_rows"] == len(df)
    # the tile-level tree is readable by load_features through the tile index as well
    et.use_tile_index(tdir)
    try:
        Xt, dt = extract.load_features("fake", et.TILE_DATASET, root=str(tmp_path / "ft"))
    finally:
        extract.load_index = load_index
    assert np.array_equal(Xt, feats["official"]) and dt.path.tolist() == tiles.path.tolist()


def test_pooling_idempotent_and_guarded(tmp_path, capsys):
    df, tiles, tdir, _ = _fake_tree(tmp_path, 2)
    pooled = str(tmp_path / "pooled")
    et.pool("fake", tdir, str(tmp_path / "ft"), pooled)
    shard = extract.shard_path(os.path.join(pooled, "fake", "bracs"), 0)
    mtime = os.path.getmtime(shard)
    et.pool("fake", tdir, str(tmp_path / "ft"), pooled)
    assert "[skip]" in capsys.readouterr().out and os.path.getmtime(shard) == mtime
    # crash before DONE: rerun completes
    os.remove(os.path.join(pooled, "fake", "bracs", "DONE"))
    os.remove(shard)
    et.pool("fake", tdir, str(tmp_path / "ft"), pooled)
    assert extract.load_features("fake", "bracs", root=pooled)[0].shape[0] == len(df)
    # tile features without DONE are refused
    os.remove(os.path.join(str(tmp_path / "ft"), "fake", et.TILE_DATASET, "DONE"))
    with pytest.raises(RuntimeError, match="incomplete"):
        et.pool("fake", tdir, str(tmp_path / "ft"), str(tmp_path / "pooled2"))


def test_pooling_refuses_roi_without_tiles(tmp_path):
    n = np.ones(len(load_index("bracs")), int)
    n[5] = 0
    _, _, tdir, _ = _fake_tree(tmp_path, n)
    with pytest.raises(RuntimeError, match="without tiles"):
        et.pool("fake", tdir, str(tmp_path / "ft"), str(tmp_path / "pooled"))


# ---------------------------------------------------------------- GPU smoke test on real tiles

@pytest.mark.gpu
def test_smoke_extract_real_tiles_phikon(tmp_path):
    if not os.path.exists(os.path.join(tb.OUT_DIR, "DONE")):
        pytest.skip("real tiling not finished")
    scratch = os.environ.get("VFM_SMOKE_DIR", str(tmp_path))
    out = os.path.join(scratch, "smoke_tiles_phikon")
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, "extract_tiles.py"), "--models", "phikon", "--subset", "50",
           "--shard-size", "16", "--batch-size", "16", "--workers", "2", "--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=CODE_DIR)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    et.use_tile_index(tb.OUT_DIR)
    try:
        X, d = extract.load_features("phikon", et.TILE_DATASET, root=out)
        # resume: lose a shard and DONE, rerun -> identical features
        fdir = os.path.join(out, "phikon", et.TILE_DATASET)
        os.remove(os.path.join(fdir, "DONE"))
        os.remove(extract.shard_path(fdir, 1))
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=CODE_DIR)
        assert p.returncode == 0 and "1 pending" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
        X2, _ = extract.load_features("phikon", et.TILE_DATASET, root=out)
    finally:
        extract.load_index = load_index
    assert X.shape == (50, 768) and np.isfinite(X).all() and np.array_equal(X, X2)
    # features equal a direct forward pass of the model on the tile PNG with its own transform
    from common import set_strict_fp32
    from models import MODELS
    set_strict_fp32()
    ext, tf, _ = MODELS["phikon"]()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = ext.to(dev)
    x = torch.stack([tf(Image.open(os.path.join(BASE_DIR, pth)).convert("RGB")) for pth in d.path[:4]])
    with torch.inference_mode():
        ref = ext(x.to(dev))["official"].cpu().numpy()
    np.testing.assert_allclose(X[:4], ref, rtol=1e-4, atol=1e-4)
    meta = json.load(open(os.path.join(out, "phikon", et.TILE_DATASET, "meta.json")))
    assert meta["config"]["model_meta"]["tiles"]["tiling_config"]["target_mpp"] == 0.5
