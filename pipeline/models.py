"""Frozen feature extractors, one per model, each following the model's official recipe.

Every loader
  * pins the checkpoint (HF revision or local file sha256),
  * loads weights strictly (any missing/unexpected backbone tensor raises),
  * returns the model in eval() mode, FP32, with no gradients,
  * returns a transform built from the model's published preprocessing.

extractor(x) returns a dict of float32 tensors:
  official    the embedding recommended by the model authors (used for the main benchmark)
  cls         final-layer class token, after the final norm (ViTs)
  patch_mean  mean of final-layer patch tokens, excluding class and register tokens (ViTs)
cls/patch_mean are used for the embedding sensitivity analysis.

Geometry: every model sees Resize(shorter side -> input size) + CenterCrop(input size). For square
tiles this is a plain resize; BRACS RoIs are first resized whole to a square (extract.WHOLE_IMAGE), so
nothing is cropped.
Exception: Prov-GigaPath's card resizes to 256 and centre-crops 224 (keeps the central 87.5%).
"""
import os

import torch
import torch.nn as nn
from torchvision import transforms

from common import WEIGHTS_DIR, sha256_file

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

REVISIONS = {
    "MahmoodLab/UNI2-h": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
    "paige-ai/Virchow2": "3158645804b69e3f3bc4439d4116edddf0840a72",
    "MahmoodLab/conch": "f9ca9f877171a28ade80228fb195ac5d79003357",
    "owkin/phikon": "057cc0295895c2df3dd7681a89680da6015cbefe",
    "timm/vit_base_patch16_224.augreg_in1k": "458542882691a06a8b667c6fb5fe5c9573093a81",
    "bioptimus/H-optimus-0": "b145cc1e6c6b30d3251aa8b1f844e6974188a743",
    "prov-gigapath/prov-gigapath": "64f9e26c15019f2d4f6d9113c6822f88bb16b01b",
}
HOPTIMUS_MEAN, HOPTIMUS_STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)


def _transform(size, mean, std, interpolation, resize=None):
    # resize (shorter side) defaults to size; only Prov-GigaPath's card resizes larger (256) before the 224 crop.
    return transforms.Compose([
        transforms.Resize(size if resize is None else resize, interpolation=interpolation),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _hf_file(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, revision=REVISIONS[repo])


def _load_strict(model, state_dict, name):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: weights do not match the architecture. "
                           f"missing={missing[:10]} ({len(missing)}) unexpected={unexpected[:10]} ({len(unexpected)})")


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _TimmViT(nn.Module):
    def __init__(self, model, official):
        super().__init__()
        self.model, self.official, self.n_prefix = model, official, model.num_prefix_tokens

    def forward(self, x):
        tokens = self.model.forward_features(x)  # includes the final norm
        cls, patch_mean = tokens[:, 0], tokens[:, self.n_prefix:].mean(dim=1)
        official = cls if self.official == "cls" else torch.cat([cls, patch_mean], dim=-1)
        return {"official": official, "cls": cls, "patch_mean": patch_mean}


def load_uni2h():
    # Model card: timm vit_giant_patch14_224 with these kwargs; embedding = class token (global_pool='token').
    import timm
    repo = "MahmoodLab/UNI2-h"
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    _load_strict(model, torch.load(_hf_file(repo, "pytorch_model.bin"), map_location="cpu"), repo)
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="cls (1536)", input=224,
                norm="imagenet", interpolation="bilinear")
    return _freeze(_TimmViT(model, "cls")), _transform(224, IMAGENET_MEAN, IMAGENET_STD,
                                                        transforms.InterpolationMode.BILINEAR), meta


def load_virchow2():
    # Model card: embedding = concat(class token, mean of patch tokens), registers (tokens 1-4) excluded.
    import timm
    repo = "paige-ai/Virchow2"
    kw = dict(img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375, global_pool="",
              dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU)
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, **kw)
    _load_strict(model, torch.load(_hf_file(repo, "pytorch_model.bin"), map_location="cpu"), repo)
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="concat(cls, mean patch tokens) (2560)",
                input=224, norm="imagenet", interpolation="bicubic")
    return _freeze(_TimmViT(model, "cls+patch_mean")), _transform(224, IMAGENET_MEAN, IMAGENET_STD,
                                                                   transforms.InterpolationMode.BICUBIC), meta


class _Phikon(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        tokens = self.vit(pixel_values=x).last_hidden_state
        cls = tokens[:, 0]
        return {"official": cls, "cls": cls, "patch_mean": tokens[:, 1:].mean(dim=1)}


def load_phikon():
    # Model card: ViTModel last_hidden_state[:, 0] (768). Preprocessing from the pinned preprocessor_config.json.
    import json
    from transformers import ViTModel
    repo = "owkin/phikon"
    pp = json.load(open(_hf_file(repo, "preprocessor_config.json")))
    vit = ViTModel.from_pretrained(repo, revision=REVISIONS[repo], add_pooling_layer=False)
    # The config sets no resample, so ViTImageProcessor's default (2 = bilinear) applies.
    interp = {2: transforms.InterpolationMode.BILINEAR, 3: transforms.InterpolationMode.BICUBIC}[pp.get("resample", 2)]
    size = pp["size"]["height"] if isinstance(pp["size"], dict) else pp["size"]
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="cls (768)", input=size,
                norm=f"mean={pp['image_mean']} std={pp['image_std']}", interpolation=interp.value)
    return _freeze(_Phikon(vit)), _transform(size, tuple(pp["image_mean"]), tuple(pp["image_std"]), interp), meta


class _Conch(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        tokens = self.visual.trunk(x, **self.visual.trunk_kwargs)
        # Same computation as CoCa.encode_image(x, proj_contrast=False, normalize=False), run once on shared tokens.
        pooled = self.visual.ln_contrast(self.visual.attn_pool_contrast(tokens)[:, 0])
        return {"official": pooled, "cls": tokens[:, 0], "patch_mean": tokens[:, 1:].mean(dim=1)}


def load_conch():
    # Official package (github.com/Mahmoodlab/CONCH). Model card for linear probing:
    # encode_image(img, proj_contrast=False, normalize=False) -> 512-d; input 448, OpenAI CLIP normalization.
    from conch.open_clip_custom.factory import create_model, read_state_dict
    repo = "MahmoodLab/conch"
    ckpt = _hf_file(repo, "pytorch_model.bin")
    model = create_model("conch_ViT-B-16", checkpoint_path=ckpt)  # loads with strict=False internally
    sd = read_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if [k for k in missing + unexpected if k.startswith("visual.")]:
        raise RuntimeError(f"{repo}: visual tower mismatch missing={missing} unexpected={unexpected}")
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="attn-pooled, pre-projection (512)",
                input=448, norm="openai_clip", interpolation="bicubic",
                text_tower_mismatch=len(missing) + len(unexpected))
    return _freeze(_Conch(model.visual)), _transform(448, OPENAI_MEAN, OPENAI_STD,
                                                      transforms.InterpolationMode.BICUBIC), meta


class _ConvStem(nn.Module):
    """TransPath ConvStem (github.com/Xiyue-Wang/TransPath), NHWC output for timm>=0.9 Swin."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        assert patch_size == 4 and embed_dim % 8 == 0
        img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.img_size, self.patch_size = tuple(img_size), (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)  # read by timm's SwinTransformer
        stem, cin, cout = [], in_chans, embed_dim // 8
        for _ in range(2):
            stem += [nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            cin, cout = cout, cout * 2
        stem.append(nn.Conv2d(cin, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class _CTransPath(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin

    def forward(self, x):
        pooled = self.swin(x)  # final norm + global average pool (TransPath: head = Identity)
        return {"official": pooled}


def load_ctranspath():
    # TransPath release checkpoint (timm 0.5.4 layout). timm>=0.9 moved PatchMerging from the end of
    # stage i to the start of stage i+1, so layers.i.downsample -> layers.(i+1).downsample.
    import re
    import timm
    path = os.path.join(WEIGHTS_DIR, "ctranspath.pth")
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, embed_layer=_ConvStem, num_classes=0)
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    out = {}
    for k, v in sd.items():
        if "relative_position_index" in k or "attn_mask" in k or k.startswith("head."):
            continue
        out[re.sub(r"layers\.(\d+)\.downsample", lambda m: f"layers.{int(m.group(1)) + 1}.downsample", k)] = v
    _load_strict(model, out, "ctranspath")
    meta = dict(checkpoint="ctranspath.pth (TransPath release)", sha256=sha256_file(path),
                official="global average pool (768)", input=224, norm="imagenet", interpolation="bilinear")
    return _freeze(_CTransPath(model)), _transform(224, IMAGENET_MEAN, IMAGENET_STD,
                                                    transforms.InterpolationMode.BILINEAR), meta


def load_vit_in1k():
    # ImageNet-1k-only baseline (AugReg ViT-B/16). Normalization mean=std=0.5 from its pretrained config.
    import timm
    from safetensors.torch import load_file
    repo = "timm/vit_base_patch16_224.augreg_in1k"
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    sd = {k: v for k, v in load_file(_hf_file(repo, "model.safetensors")).items() if not k.startswith("head.")}
    _load_strict(model, sd, repo)
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="cls (768)", input=224,
                norm="mean=std=0.5", interpolation="bicubic")
    return _freeze(_TimmViT(model, "cls")), _transform(224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                        transforms.InterpolationMode.BICUBIC), meta


def load_hoptimus0():
    # Model card: timm.create_model("hf-hub:bioptimus/H-optimus-0", init_values=1e-5, dynamic_img_size=False);
    # config.json: vit_giant_patch14_reg4_dinov2, fixed input 224, global_pool='token'. Embedding = model(x) =
    # class token after the final norm (1536). Card transform: ToTensor + Normalize(card mean/std) on 224x224
    # tiles at 0.5 mpp; it has no resize step, so other sizes get Resize + CenterCrop with timm's default
    # interpolation for this config (bicubic; the card and config give none). Card suggests fp16 autocast: we run FP32.
    import timm
    repo = "bioptimus/H-optimus-0"
    model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, img_size=224, init_values=1e-5,
                              dynamic_img_size=False, num_classes=0)
    _load_strict(model, torch.load(_hf_file(repo, "pytorch_model.bin"), map_location="cpu"), repo)
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="cls (1536)", input=224,
                norm=f"mean={list(HOPTIMUS_MEAN)} std={list(HOPTIMUS_STD)}", interpolation="bicubic")
    return _freeze(_TimmViT(model, "cls")), _transform(224, HOPTIMUS_MEAN, HOPTIMUS_STD,
                                                        transforms.InterpolationMode.BICUBIC), meta


def load_gigapath():
    # Prov-GigaPath tile encoder. Model card: timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True);
    # embedding = tile_encoder(x) = class token after the final norm (global_pool='token'), 1536. Architecture =
    # config.json (vit_giant_patch14_dinov2 with its model_args: patch 16, depth 40, mlp_ratio 5.33334).
    # Card transform: Resize(256, bicubic) + CenterCrop(224) + ImageNet normalization. (config.json's
    # pretrained_cfg says crop_pct=1.0, i.e. no crop; we follow the card's explicit recipe.)
    import json
    import timm
    repo = "prov-gigapath/prov-gigapath"
    cfg = json.load(open(_hf_file(repo, "config.json")))
    assert cfg["architecture"] == "vit_giant_patch14_dinov2" and cfg["global_pool"] == "token", cfg
    model = timm.create_model(cfg["architecture"], pretrained=False, **cfg["model_args"])
    _load_strict(model, torch.load(_hf_file(repo, "pytorch_model.bin"), map_location="cpu"), repo)
    meta = dict(checkpoint=repo, revision=REVISIONS[repo], official="cls (1536)", input=224, resize=256,
                norm="imagenet", interpolation="bicubic")
    return _freeze(_TimmViT(model, "cls")), _transform(224, IMAGENET_MEAN, IMAGENET_STD,
                                                        transforms.InterpolationMode.BICUBIC, resize=256), meta


MODELS = {
    "uni2h": load_uni2h, "virchow2": load_virchow2, "phikon": load_phikon, "conch": load_conch,
    "ctranspath": load_ctranspath, "vit_in1k": load_vit_in1k,
    "hoptimus0": load_hoptimus0, "gigapath": load_gigapath,
}
