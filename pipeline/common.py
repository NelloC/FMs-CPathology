"""Shared paths, atomic I/O and environment logging."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

# Code root = parent of pipeline/ (holds indexes/, features/, verification/).
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_data_root(start):
    """Nearest ancestor of start that contains dataset/."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "dataset")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return start
        d = parent


# Data root = the directory containing dataset/; index paths are relative to it. Override with VFM_BASE_DIR.
BASE_DIR = os.environ.get("VFM_BASE_DIR", _find_data_root(CODE_DIR))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
INDEX_DIR = os.environ.get("VFM_INDEX_DIR", os.path.join(CODE_DIR, "indexes"))
FEATURES_DIR = os.environ.get("VFM_FEATURES_DIR", os.path.join(CODE_DIR, "features"))
WEIGHTS_DIR = os.environ.get("VFM_WEIGHTS_DIR", os.path.join(CODE_DIR, "weights"))


def atomic_write_bytes(path, data):
    """Write to a temp file in the same directory, then rename: a crash never leaves a partial file."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_strings(items):
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


def set_strict_fp32():
    """True FP32: disable TF32 for matmul and cuDNN (PyTorch enables TF32 for cuDNN convs by default)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def environment_info():
    import numpy
    import PIL
    import torch
    import torchvision
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for mod in ["timm", "transformers", "open_clip", "safetensors", "sklearn", "pandas"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        from importlib.metadata import distribution
        direct = distribution("conch").read_text("direct_url.json")
        info["conch"] = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else "installed"
    except Exception:
        info["conch"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["nvidia_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            info["nvidia_driver"] = None
    return info
