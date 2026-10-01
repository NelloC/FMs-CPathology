"""pipeline/precision.py on real Phikon features (functional check; timings are not meaningful here)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from common import CODE_DIR, FEATURES_DIR
from conftest import PIPELINE_DIR

pytestmark = pytest.mark.gpu


def test_precision_phikon_lung(tmp_path):
    if not os.path.exists(os.path.join(FEATURES_DIR, "phikon", "lung", "DONE")) or \
            not os.path.exists(os.path.join(CODE_DIR, "results", "probe", "phikon", "lung", "seed42.json")):
        pytest.skip("phikon/lung features or probe not available yet")
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, "precision.py"), "--models", "phikon", "--datasets", "lung",
           "--n-images", "96", "--out", str(tmp_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    d = tmp_path / "phikon" / "lung"
    res = {k: json.load(open(d / f"{k}.json")) for k in ["FP32_GPU", "FP16_GPU", "BF16_GPU", "FP32_CPU", "INT8_CPU"]}

    # FP32 GPU features = the cached extraction features of the same images
    from extract import load_features
    X, _ = load_features("phikon", "lung")
    ref = np.load(d / "FP32_GPU.npz")
    assert np.allclose(ref["feats"], X[ref["rows"]], atol=1e-4)
    # FP32 CPU ~ FP32 GPU; reduced precisions close to FP32
    assert res["FP32_CPU"]["vs_fp32_gpu"]["cosine_min"] > 0.9999
    for k in ["FP16_GPU", "BF16_GPU", "INT8_CPU"]:
        assert res[k]["vs_fp32_gpu"]["cosine_mean"] > 0.95, (k, res[k]["vs_fp32_gpu"])
    for r in res.values():
        t = r["timing"]
        assert t["n_images_timed"] == 96 and t["ms_per_image_mean"] > 0 and len(t["times_s"]) == 3
    print({k: (round(v["timing"]["ms_per_image_mean"], 2), v.get("vs_fp32_gpu")) for k, v in res.items()})
    # rerun skips
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert "[skip] phikon/lung" in p.stdout
