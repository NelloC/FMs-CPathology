import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import timm
import gc
import json

MODELS_TO_RUN = ["phikon","ctranspath"]
BATCH_SIZE = 8       
NUM_WORKERS = 0       

SAVE_DIR = "./results_quantization_final_3"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_DIR = "/Users/aconelli/TechConnect/FundationalModels/dataset"
DIRS = {
    "TCGA_LUNG": f"{BASE_DIR}/TGCA/lung_colon_image_set/Train and Validation Set",
    "BREAKHIS": f"{BASE_DIR}/BreakHis - Breast Cancer Histopathological Database/dataset_cancer_v1/dataset_cancer_v1/classificacao_binaria",
    "NCT_COLON": f"{BASE_DIR}/NCT-CRC-HE-100K",
    "HUBMAP": f"{BASE_DIR}/HUBMAP_TILED_TIFF",
    "BACH": f"{BASE_DIR}/BACH/TILED_TIFF"
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
cpu_device = torch.device("cpu")
print(f"STARTING QUANTIZATION BENCHMARK (FP32 -> FP16 -> INT8) | Device: {device}")

class GenericDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.filepaths[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
        except: return torch.zeros(3, 224, 224), 0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_val_loader(paths, labels):
    if len(paths) == 0: return None, 0
    _, v_p, _, v_l = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    v_loader = DataLoader(GenericDataset(v_p, v_l, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return v_loader, len(v_p)

def scan_generic(base_dir):
    p, l = [], []
    if not os.path.exists(base_dir): return p, l, 0
    c_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    c_map = {n: i for i, n in enumerate(c_names)}
    for cn, ci in c_map.items():
        for r, _, fs in os.walk(os.path.join(base_dir, cn)):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    p.append(os.path.join(r, f)); l.append(ci)
    return p, l, len(c_names)

task_loaders, task_classes = {}, {}
print("Scanning Data (Loading for Inference only)...")

p, l = [], []
for r, _, fs in os.walk(DIRS["TCGA_LUNG"]):
    for f in fs:
        if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.jpeg')):
            if 'lung_aca' in r.lower(): p.append(os.path.join(r, f)); l.append(0)
            elif 'lung_scc' in r.lower(): p.append(os.path.join(r, f)); l.append(1)
if p: task_loaders["lung"], task_classes["lung"] = create_val_loader(p, l)[0], 2

p, l = [], []
for r, _, fs in os.walk(DIRS["BREAKHIS"]):
    for f in fs:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            if 'benign' in r.lower(): p.append(os.path.join(r, f)); l.append(0)
            elif 'malignant' in r.lower(): p.append(os.path.join(r, f)); l.append(1)
if p: task_loaders["breakhis"], task_classes["breakhis"] = create_val_loader(p, l)[0], 2

for task_name, dir_key in {"nct": "NCT_COLON", "hubmap": "HUBMAP", "bach": "BACH"}.items():
    p, l, num_c = scan_generic(DIRS[dir_key])
    if p and num_c > 0: task_loaders[task_name], task_classes[task_name] = create_val_loader(p, l)[0], num_c

active_tasks = list(task_classes.keys())
print(f"DATASETS READY: {active_tasks}")

class UniversalPanCancerModel(nn.Module):
    def __init__(self, backbone, model_name, task_classes_dict, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        for param in self.parameters(): param.requires_grad = False
        self.heads = nn.ModuleDict({task: nn.Linear(emb_dim, num_c) for task, num_c in task_classes_dict.items()})

    def forward(self, x, task):
        if self.model_name == "ctranspath": feat = self.backbone.forward_features(x).mean(dim=[1, 2])
        elif self.model_name == "conch":
            feat = self.backbone(x)
            if isinstance(feat, tuple): feat = feat[0]
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)): feat = feat[0]
            if feat.dim() == 3: feat = feat.mean(dim=1)
            elif feat.dim() == 4: feat = feat.mean(dim=[2, 3])
        return self.heads[task](feat)

def run_benchmark(model, loader, task_name, num_classes, quant_level, test_device):
    model.eval() 
    all_preds, all_targets, all_probs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(test_device), lbls.to(test_device).long()
            if quant_level == "FP16": imgs = imgs.half()
            outs = model(imgs, task=task_name)

            probs = torch.softmax(outs.float(), dim=1) 

            _, preds = torch.max(outs, 1)
            all_preds.extend(preds.cpu().numpy()); all_targets.extend(lbls.cpu().numpy())
            if num_classes == 2: all_probs.extend(probs[:, 1].cpu().float().numpy())
            else: all_probs.extend(probs.cpu().float().numpy())

    total_time = time.time() - start_time
    acc = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    try: auc = roc_auc_score(all_targets, all_probs) if num_classes == 2 else roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except: auc = 0.5
    return {"acc": acc, "auc": auc, "f1": f1, "time_seconds": total_time}

def load_weights_safely(model, current_model_name, path):
    state_dict = torch.load(path, map_location="cpu")
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        if current_model_name == "phikon" and new_key.startswith("backbone.") and not new_key.startswith("backbone.vit."):
            new_key = new_key.replace("backbone.", "backbone.vit.")
            
        elif current_model_name == "ctranspath" and "patch_embed" in new_key:
            new_key = new_key.replace("patch_embed.proj.0", "patch_embed.proj.0") 
            
        new_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"[DEBUG {current_model_name.upper()}] Weights aligned! Missing keys: {len(missing)}")
    return model

for current_model in MODELS_TO_RUN:
    print(f"\n{'='*70}\nSTARTING QUANTIZATION: {current_model.upper()}\n{'='*70}")

    PATH_MODELLO = f"/Users/aconelli/TechConnect/FundationalModels/multiple_learning/risultati_finali_pancancer/{current_model}/PanCancer_{current_model}_Epoch_10.pth"
    
    if not os.path.exists(PATH_MODELLO):
        print(f"Weights for {current_model.upper()} not found at path:\n {PATH_MODELLO}\n Skipping to next model.")
        continue

    if current_model == "uni":
        from timm.layers import SwiGLUPacked
        timm_kwargs = {'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True, 'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True}
        backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=False, **timm_kwargs)
        embed_dim = 1536
    elif current_model == "virchow2":
        from timm.layers import SwiGLUPacked
        backbone = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, num_classes=0)
        embed_dim = 1280
    elif current_model == "phikon":
            backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
            embed_dim = 768
    elif current_model == "conch":
        import open_clip
        from open_clip import factory
        factory._MODEL_CONFIGS['conch_ViT-B-16'] = {
            "embed_dim": 512, "vision_cfg": {"image_size": 224, "layers": 12, "width": 768, "patch_size": 16},
            "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12}
        }
        full_model, _, _ = open_clip.create_model_and_transforms('conch_ViT-B-16', pretrained=None)
        backbone = full_model.visual
        embed_dim = 512
    elif current_model == "ctranspath":
        class ConvStem(nn.Module):
            def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
                super().__init__()
                self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(embed_dim // 2), nn.ReLU(inplace=True), nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
            def forward(self, x): return self.proj(x).permute(0, 2, 3, 1)
        backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        backbone.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=128, norm_layer=nn.LayerNorm)
        embed_dim = 1024

    results_dict = {task: {} for task in active_tasks}

    for task in active_tasks:
        print(f"\nTASK: {task.upper()}")
        
        print("   Testing FP32...")
        base_model = UniversalPanCancerModel(backbone, current_model, task_classes, embed_dim)
        base_model = load_weights_safely(base_model, current_model, PATH_MODELLO)
        base_model = base_model.to(device)
        state_dict = torch.load(PATH_MODELLO, map_location="cpu")
        missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
        print(f"[DEBUG {current_model}] Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
        base_model = base_model.to(device) 
        res_32 = run_benchmark(base_model, task_loaders[task], task, task_classes[task], "FP32", device)
        results_dict[task]["FP32"] = res_32
        print(f"      FP32 | Acc: {res_32['acc']:.1f}% | AUC: {res_32['auc']:.4f} | Time: {res_32['time_seconds']:.2f}s")

        print("   Testing FP16...")
        fp16_model = base_model.half()
        res_16 = run_benchmark(fp16_model, task_loaders[task], task, task_classes[task], "FP16", device)
        results_dict[task]["FP16"] = res_16
        print(f"      FP16 | Acc: {res_16['acc']:.1f}% | AUC: {res_16['auc']:.4f} | Time: {res_16['time_seconds']:.2f}s")

        del fp16_model, base_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        print("   Testing INT8 (CPU)...")
        backbone = backbone.float()
        cpu_model = UniversalPanCancerModel(backbone, current_model, task_classes, embed_dim)
        cpu_model = load_weights_safely(cpu_model, current_model, PATH_MODELLO)
        cpu_model = cpu_model.to(cpu_device)
        
        torch.backends.quantized.engine = 'qnnpack'
        
        int8_model = torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8) 
        res_8 = run_benchmark(int8_model, task_loaders[task], task, task_classes[task], "INT8", cpu_device)
        results_dict[task]["INT8"] = res_8
        print(f"      INT8 | Acc: {res_8['acc']:.1f}% | AUC: {res_8['auc']:.4f} | Time: {res_8['time_seconds']:.2f}s")
        del cpu_model, int8_model; gc.collect()

    json_path = os.path.join(SAVE_DIR, f"{current_model}_quantization.json")
    with open(json_path, "w") as f: json.dump(results_dict, f, indent=4)
    print(f"Results saved: {json_path}")

    del backbone; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

print("\nALL QUANTIZATION BENCHMARKS COMPLETED!")
