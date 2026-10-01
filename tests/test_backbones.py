"""Every backbone loads strictly from its pinned checkpoint, stays frozen in eval mode, and matches the
authors' reference code."""
import json
import os

import pytest
import torch
from PIL import Image

from common import BASE_DIR, CODE_DIR, set_strict_fp32
from data_index import load_index
from models import MODELS

pytestmark = pytest.mark.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_verification_report():
    """verify_models.py compared each loader with the authors' reference code on an NCT (square), a SICAP
    and a BRACS image. Square tiles must match to float precision; Phikon's reference resizes
    non-square images without cropping, so it differs on BRACS."""
    report = json.load(open(os.path.join(CODE_DIR, "verification", "models.json")))["models"]
    assert set(report) == {"uni2h", "virchow2", "phikon", "conch", "ctranspath", "vit_in1k", "hoptimus0", "gigapath"}
    for name, r in report.items():
        assert r["finite"], name
        if "max_abs_diff_official" in r:  # ctranspath has no separate reference implementation
            assert r["max_abs_diff_official"][0] < 1e-4, (name, r["max_abs_diff_official"])
    # The model cards' own transforms apply to every image for Prov-GigaPath (all three must match); the
    # H-optimus-0 card has no resize step, so its reference uses the pipeline geometry for non-224 images.
    for name in ["hoptimus0", "gigapath"]:
        assert max(report[name]["max_abs_diff_input"]) == 0.0, name
        assert max(report[name]["max_abs_diff_official"]) < 1e-4, (name, report[name]["max_abs_diff_official"])


@pytest.mark.parametrize("name,repo", [("hoptimus0", "bioptimus/H-optimus-0"), ("gigapath", "prov-gigapath/prov-gigapath")])
def test_new_backbone_strict_and_reference(name, repo):
    """Every checkpoint tensor is in the model with its checkpoint value, a missing tensor raises, and the
    'official' embedding equals the model card's reference code on a square NCT tile (CPU, FP32)."""
    import verify_models
    from models import _hf_file, _load_strict
    set_strict_fp32()
    extractor, transform, meta = MODELS[name]()
    assert meta["revision"] and not extractor.training
    assert not any(p.requires_grad for p in extractor.parameters())
    sd = torch.load(_hf_file(repo, "pytorch_model.bin"), map_location="cpu")
    msd = extractor.model.state_dict()
    assert set(sd) == set(msd)
    assert all(torch.equal(sd[k], msd[k]) for k in sd)
    with pytest.raises(RuntimeError, match="missing"):
        _load_strict(extractor.model, {k: v for k, v in sd.items() if k != "norm.weight"}, name)
    del sd, msd

    img = Image.open(os.path.join(BASE_DIR, load_index("nct").path[0])).convert("RGB")
    ref_t, ref_emb = verify_models.reference(name)
    x, x_ref = transform(img).unsqueeze(0), ref_t(img).unsqueeze(0)
    assert torch.equal(x, x_ref)
    with torch.inference_mode():
        out, e_ref = extractor(x), ref_emb(x_ref)
    assert out["official"].shape == (1, 1536) and out["official"].dtype == torch.float32
    assert torch.allclose(out["official"], e_ref, rtol=0, atol=1e-4), (out["official"] - e_ref).abs().max()
