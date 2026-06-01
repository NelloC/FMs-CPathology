import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import timm
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
import time
import json
from huggingface_hub import login

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
elif torch.backends.mps.is_available(): torch.mps.empty_cache()
Image.MAX_IMAGE_PIXELS = None



MODELS_TO_RUN = ["uni", "virchow2","phikon", "conch", "ctranspath"]
QUANTIZATION_LEVEL = "FP32"
BATCH_SIZE = 8
NUM_WORKERS = 0 
EPOCHS = 10 
LR = 1e-4

SUBSAMPLE_FRAC = {
    "panda": 0.25,
    "bracs": 1.0,
    "sicap": 1.0
}

DIRS = {
    "PANDA": "/Users/aconelli/TechConnect/FundationalModels/dataset/PANDA_TILED_TIFF", 
    "BRACS": "/Users/aconelli/TechConnect/FundationalModels/dataset/BRACS/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/train",
    "SICAP": "/Users/aconelli/TechConnect/FundationalModels/dataset/SICAPv2_Formatted"
}

task_mapping = {
    "panda": "PANDA",
    "bracs": "BRACS",
    "sicap": "SICAP"
}

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

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
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_loader(paths, labels):
    if len(paths) == 0: return None, None, 0
    t_p, v_p, t_l, v_l = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    t_loader = DataLoader(GenericDataset(t_p, t_l, transform), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    v_loader = DataLoader(GenericDataset(v_p, v_l, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return t_loader, v_loader, len(t_p)

class UniversalPanCancerModel(nn.Module):
    def __init__(self, backbone, model_name, task_classes_dict, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.heads = nn.ModuleDict()
        for task, num_c in task_classes_dict.items():
            self.heads[task] = nn.Linear(emb_dim, num_c)

    def forward(self, x, task):
        if self.model_name == "ctranspath":
            feat = self.backbone.forward_features(x).mean(dim=[1, 2])
        elif self.model_name == "conch":
            feat = self.backbone(x)
            if isinstance(feat, tuple): feat = feat[0]
        else: 
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)): feat = feat[0]
            if feat.dim() == 3: feat = feat.mean(dim=1)
            elif feat.dim() == 4: feat = feat.mean(dim=[2, 3])

        feat = feat.float()    
        return self.heads[task](feat)

def evaluate_clinical(model, loader, task_name, num_classes):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device).long()
            if QUANTIZATION_LEVEL == "FP16": imgs = imgs.half()

            outs = model(imgs, task=task_name)
            probs = torch.softmax(outs, dim=1)
            _, preds = torch.max(outs, 1)

            all_preds.extend(preds.cpu().numpy()); all_targets.extend(lbls.cpu().numpy())
            if num_classes == 2: all_probs.extend(probs[:, 1].cpu().float().numpy())
            else: all_probs.extend(probs.cpu().float().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic') 

    try:
        if num_classes == 2: auc = roc_auc_score(all_targets, all_probs)
        else: auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except: auc = 0.5

    return {"acc": acc * 100, "auc": auc, "f1": f1, "qwk": qwk}


if __name__ == '__main__':
    print(f"STARTING ORDINAL BENCHMARK (V2) | QUANTIZATION: {QUANTIZATION_LEVEL} | Device: {device}")
    
    task_loaders, task_classes = {}, {}
    active_tasks = []
    print("\nScanning Data (Auto-Discovery + Subsampling Mode)...")

    for task_name, dir_key in task_mapping.items():
        base_dir = DIRS[dir_key]
        try: subfolders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        except FileNotFoundError:
            print(f"Directory not found for {task_name.upper()}, skipping.")
            continue

        class_to_idx = {folder_name: i for i, folder_name in enumerate(subfolders)}
        num_classes = len(class_to_idx)
        if num_classes == 0: continue

        paths, labels = [], []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    for folder_name, idx in class_to_idx.items():
                        if f"/{folder_name}/" in f"/{root}/" or root.endswith(folder_name):
                            paths.append(os.path.join(root, f))
                            labels.append(idx)
                            break

        if len(paths) > 0:
            frac = SUBSAMPLE_FRAC.get(task_name, 1.0)
            if frac < 1.0:
                print(f"   Reducing {task_name.upper()} to {frac*100}% to speed up training...")
                paths, _, labels, _ = train_test_split(
                    paths, labels, train_size=frac, random_state=42, stratify=labels
                )

            t_loader, v_loader, num_train = create_loader(paths, labels)
            task_loaders[task_name] = t_loader
            task_loaders[f"{task_name}_val"] = v_loader
            task_classes[task_name] = num_classes
            active_tasks.append(task_name)
            print(f"{task_name.upper():<8} ready: {num_train} train img | {num_classes} classes detected.")

    print(f"\nACTIVE TASKS: {active_tasks}")
    print("-" * 60)

    for current_model_name in MODELS_TO_RUN:
        print(f"\n" + "="*60)
        print(f"INITIALIZING MODEL: {current_model_name.upper()}")
        print("="*60)

        SAVE_DIR = f"./risultati_ordinali_v3/{current_model_name}_{QUANTIZATION_LEVEL}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        if current_model_name == "uni":
            from timm.layers import SwiGLUPacked
            timm_kwargs = {'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True, 'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True}
            backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            embed_dim = 1536
        elif current_model_name == "virchow2":
            from timm.layers import SwiGLUPacked
            backbone = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, num_classes=0)
            embed_dim = 1280
        elif current_model_name == "phikon":
            backbone = timm.create_model("hf-hub:owkin/phikon", pretrained=True, num_classes=0)
            embed_dim = 768
        elif current_model_name == "conch":
            import open_clip
            from open_clip import factory
            from huggingface_hub import hf_hub_download
            
            factory._MODEL_CONFIGS['conch_ViT-B-16'] = {
                "embed_dim": 512, "vision_cfg": {"image_size": 224, "layers": 12, "width": 768, "patch_size": 16},
                "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12}
            }
            print("Downloading/Verifying CONCH weights...")
            conch_weights_path = hf_hub_download(repo_id="MahmoodLab/conch", filename="pytorch_model.bin")
            
            base_model, _, _ = open_clip.create_model_and_transforms('conch_ViT-B-16', pretrained=None)
            base_model.load_state_dict(torch.load(conch_weights_path, map_location="cpu"), strict=False)
            backbone = base_model.visual
            embed_dim = 512

        elif current_model_name == "ctranspath":
            def to_2tuple(x): return tuple(x) if isinstance(x, (tuple, list)) else (x, x)
            class ConvStem(nn.Module):
                def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
                    super().__init__()
                    self.proj = nn.Sequential(
                        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(embed_dim // 2), nn.ReLU(inplace=True),
                        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
                    )
                def forward(self, x): return self.proj(x).permute(0, 2, 3, 1)

            backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            backbone.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=128, norm_layer=nn.LayerNorm)
            
            CTRANSPATH_WEIGHTS = "./model_lib/pretrained/ctranspath.pth" 
            if os.path.exists(CTRANSPATH_WEIGHTS):
                ckpt = torch.load(CTRANSPATH_WEIGHTS, map_location="cpu")
                backbone.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
            else:
                print("ERROR: CTransPath weights not found!")
                
            embed_dim = 1024
            
        model = UniversalPanCancerModel(backbone, current_model_name, task_classes, embed_dim).to(device)

        if QUANTIZATION_LEVEL == "FP16":
            model = model.half()
            for task in task_classes.keys():
                model.heads[task] = model.heads[task].float() 

        params_to_train = []
        for h in model.heads.values():
            for p in h.parameters():
                p.requires_grad = True
                params_to_train.append(p)
        optimizer = optim.Adam(params_to_train, lr=LR)
        criterion = nn.CrossEntropyLoss()

        history = {task: [] for task in active_tasks}
        max_batches = max([len(task_loaders[task]) for task in active_tasks])

        for epoch in range(EPOCHS):
            start_time = time.time()
            model.train()
            iterators = {task: iter(task_loaders[task]) for task in active_tasks}

            for step in tqdm(range(max_batches), desc=f"[{current_model_name.upper()}] Epoch {epoch+1}/{EPOCHS}"):
                optimizer.zero_grad() 
                for task in active_tasks:
                    try: img, lbl = next(iterators[task])
                    except StopIteration: 
                        iterators[task] = iter(task_loaders[task])
                        img, lbl = next(iterators[task])

                    img, lbl = img.to(device), lbl.to(device).long()
                    if QUANTIZATION_LEVEL == "FP16": img = img.half()

                    loss = criterion(model(img, task=task), lbl)
                    loss.backward() 
                optimizer.step()

            print(f"\nRESULTS {current_model_name.upper()} - EPOCH {epoch+1} (Time: {time.time()-start_time:.0f}s)")
            for task in active_tasks:
                val_loader = task_loaders[f"{task}_val"]
                num_c = task_classes[task]
                metrics = evaluate_clinical(model, val_loader, task, num_c)
                history[task].append(metrics)
                print(f"{task.upper():<10} | Acc: {metrics['acc']:>4.1f}% | AUC: {metrics['auc']:.4f} | QWK: {metrics['qwk']:>7.4f}")

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{current_model_name}_{QUANTIZATION_LEVEL}_Epoch_{epoch+1}.pth"))

        with open(os.path.join(SAVE_DIR, f"{current_model_name}_{QUANTIZATION_LEVEL}_metrics.json"), "w") as f:
            json.dump(history, f, indent=4)

        print(f"Training {current_model_name.upper()} completed. Cleaning memory...")

        del model
        del backbone
        del optimizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

    print("\nPAN-CANCER ORDINAL EXPERIMENT COMPLETED SUCCESSFULLY!")
