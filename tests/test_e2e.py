"""pipeline/extract.py killed partway through and restarted with the same command must produce the same
features, bit for bit, as an uninterrupted run."""
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from conftest import PIPELINE_DIR

pytestmark = pytest.mark.gpu
PY = sys.executable
TIMING_KEYS = {"time", "environment", "time_seconds_mean", "time_seconds_std", "time_seconds_all",
               "ms_per_image", "throughput_img_per_sec", "started"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def assert_same_outputs(a, b, subdir=""):
    fa = sorted(os.path.relpath(f, a) for f in glob.glob(os.path.join(str(a), subdir, "**", "*"), recursive=True)
                if os.path.isfile(f))
    fb = sorted(os.path.relpath(f, b) for f in glob.glob(os.path.join(str(b), subdir, "**", "*"), recursive=True)
                if os.path.isfile(f))
    assert fa == fb, f"different files: {set(fa) ^ set(fb)}"
    assert fa, "no outputs"
    for rel in fa:
        pa, pb = os.path.join(str(a), rel), os.path.join(str(b), rel)
        if rel.endswith(".npz"):
            x, y = np.load(pa), np.load(pb)
            for k in x.files:
                assert np.array_equal(x[k], y[k]), f"{rel}:{k} differs"
        elif rel.endswith(".pth"):
            x, y = torch.load(pa), torch.load(pb)
            assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x), f"{rel} differs"
        elif rel.endswith(".json"):
            assert _strip(json.load(open(pa))) == _strip(json.load(open(pb))), f"{rel} differs"
        elif rel.endswith(".safetensors"):
            from safetensors.torch import load_file
            x, y = load_file(pa), load_file(pb)
            assert all(torch.equal(x[k], y[k]) for k in x), f"{rel} differs"


def test_extract_resume(tmp_path):
    """pipeline/extract.py on the real index (evenly spaced subset): killed after one shard and resumed,
    the features equal an uninterrupted extraction."""
    args = ["--models", "phikon", "--datasets", "nct", "--subset", "400", "--shard-size", "64", "--workers", "4"]
    cmd = [PY, os.path.join(PIPELINE_DIR, "extract.py")] + args
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    full, killed = tmp_path / "full", tmp_path / "killed"
    assert subprocess.run(cmd + ["--out", str(full)], env=env, capture_output=True).returncode == 0
    p = subprocess.Popen(cmd + ["--out", str(killed)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while len(glob.glob(str(killed / "phikon/nct/shards/*.safetensors"))) < 2:
        assert p.poll() is None, "finished before the kill point"
        time.sleep(0.02)
    p.send_signal(signal.SIGKILL)
    p.wait()
    assert not (killed / "phikon/nct/DONE").exists()
    assert subprocess.run(cmd + ["--out", str(killed)], env=env, capture_output=True).returncode == 0
    assert_same_outputs(full, killed)
