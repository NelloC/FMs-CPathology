"""Check each extractor against the model authors' reference code path on real images (CPU, FP32).

For every model: strict weight load (in models.py), preprocessing equal to the reference transform,
and 'official' embedding equal to the reference embedding. Updates verification/models.json in place:
entries of models not named on the command line are kept as they are.

Usage: python pipeline/verify_models.py [model ...]
"""
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, CODE_DIR, atomic_write_json, environment_info, set_strict_fp32
from data_index import load_index
from models import MODELS, REVISIONS

IMAGES = [load_index("nct").path[0], load_index("sicap").path[0], load_index("bracs").path[0]]


def reference(name):
    """Return (preprocess, embed) written as in the model cards, or None where no separate reference exists."""
    import timm
    if name == "uni2h":
        kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                  mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                  act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
        m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw).eval()
        t = timm.data.create_transform(**timm.data.resolve_data_config(m.pretrained_cfg, model=m))
        return t, lambda x: m(x)
    if name == "virchow2":
        m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU).eval()
        t = timm.data.create_transform(**timm.data.resolve_data_config(m.pretrained_cfg, model=m))

        def emb(x):
            out = m(x)
            return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)
        return t, emb
    if name == "phikon":
        from transformers import AutoImageProcessor, ViTModel
        p = AutoImageProcessor.from_pretrained("owkin/phikon", revision=REVISIONS["owkin/phikon"])
        m = ViTModel.from_pretrained("owkin/phikon", revision=REVISIONS["owkin/phikon"], add_pooling_layer=False).eval()
        return (lambda img: p(img, return_tensors="pt")["pixel_values"][0]), \
            (lambda x: m(pixel_values=x).last_hidden_state[:, 0])
    if name == "conch":
        from huggingface_hub import hf_hub_download
        from conch.open_clip_custom import create_model_from_pretrained
        ckpt = hf_hub_download("MahmoodLab/conch", "pytorch_model.bin", revision=REVISIONS["MahmoodLab/conch"])
        m, t = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt)
        m.eval()
        return t, lambda x: m.encode_image(x, proj_contrast=False, normalize=False)
    if name == "vit_in1k":
        m = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_data_config(m.pretrained_cfg, model=m)
        cfg["crop_pct"] = 1.0  # we use the whole tile instead of timm's 0.9 center crop (documented deviation)
        return timm.data.create_transform(**cfg), lambda x: m(x)
    if name == "hoptimus0":
        from torchvision import transforms
        m = timm.create_model(f"hf-hub:bioptimus/H-optimus-0@{REVISIONS['bioptimus/H-optimus-0']}", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False).eval()
        card = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ])
        # The card takes 224x224 tiles and has no resize step; other sizes (SICAP, BRACS) first get the
        # pipeline's geometry (not part of the card), so only the NCT input comparison is card-only.
        geometry = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                       transforms.CenterCrop(224)])
        return (lambda img: card(img if img.size == (224, 224) else geometry(img))), lambda x: m(x)
    if name == "gigapath":
        from torchvision import transforms
        m = timm.create_model(f"hf_hub:prov-gigapath/prov-gigapath@{REVISIONS['prov-gigapath/prov-gigapath']}",
                              pretrained=True).eval()
        t = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return t, lambda x: m(x)
    return None


def main():
    set_strict_fp32()
    torch.manual_seed(0)
    # Merge into the existing report: entries of models not re-verified in this run are kept unchanged.
    path = os.path.join(CODE_DIR, "verification", "models.json")
    env = environment_info()
    report = json.load(open(path)) if os.path.exists(path) else {"environment": env, "images": IMAGES, "models": {}}
    if report["images"] != IMAGES:
        raise RuntimeError(f"{path} was made with other images: {report['images']}")
    for name in sys.argv[1:] or list(MODELS):
        extractor, transform, meta = MODELS[name]()
        imgs = [Image.open(os.path.join(BASE_DIR, p)).convert("RGB") for p in IMAGES]
        x = torch.stack([transform(i) for i in imgs])
        with torch.inference_mode():
            out = extractor(x)
        entry = dict(meta=meta, n_params=sum(p.numel() for p in extractor.parameters()),
                     dims={k: list(v.shape[1:]) for k, v in out.items()},
                     finite=all(bool(torch.isfinite(v).all()) for v in out.values()),
                     dtype={k: str(v.dtype) for k, v in out.items()}, environment=env)
        ref = reference(name)
        if ref is not None:
            ref_t, ref_emb = ref
            x_ref = torch.stack([ref_t(i) for i in imgs])
            with torch.inference_mode():
                e_ref = ref_emb(x_ref)
            # Per image, in IMAGES order (square NCT 224, square SICAP 512, non-square BRACS RoI).
            entry["max_abs_diff_input"] = [float(d) for d in (x - x_ref).abs().amax(dim=(1, 2, 3))]
            entry["max_abs_diff_official"] = [float(d) for d in (out["official"] - e_ref).abs().amax(dim=1)]
            entry["max_abs_official"] = float(e_ref.abs().max())
        report["models"][name] = entry
        print(name, {k: v for k, v in entry.items() if k not in ("meta", "environment")}, flush=True)
        del extractor, ref
    atomic_write_json(path, report)


if __name__ == "__main__":
    main()
