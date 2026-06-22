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
import types
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import json
from huggingface_hub import login

gc.collect()
torch.cuda.empty_cache()
Image.MAX_IMAGE_PIXELS = None




MODELS_TO_RUN = ["uni", "phikon", "virchow2","conch", "ctranspath"] #

BATCH_SIZE = 8       #
NUM_WORKERS = 0
EPOCHS = 10
LR = 1e-4

BASE_DIR = "./data"

DIRS = {
    "TCGA_LUNG": f"{BASE_DIR}/TGCA/lung_colon_image_set/Train and Validation Set", 
    
    "BREAKHIS": f"{BASE_DIR}/BreakHis - Breast Cancer Histopathological Database/dataset_cancer_v1/dataset_cancer_v1/classificacao_binaria",
    

    "NCT_COLON": f"{BASE_DIR}/NCT-CRC-HE-100K",
    
    "HUBMAP": f"{BASE_DIR}/HUBMAP_TILED_TIFF",
    
    "BACH": f"{BASE_DIR}/BACH/TILED_TIFF"
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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

class UniversalPanCancerModel(nn.Module):
    def __init__(self, backbone, model_name, task_classes_dict, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.heads = nn.ModuleDict({
            task: nn.Linear(emb_dim, num_c) for task, num_c in task_classes_dict.items()
        })

    def forward(self, x, task):
        if self.model_name == "ctranspath":
            feat = self.backbone.forward_features(x).mean(dim=[1, 2])
        elif self.model_name == "conch":
            feat = self.backbone(x)
            if isinstance(feat, tuple): feat = feat[0]
        elif self.model_name == "resnet":
            feat = self.backbone(x).view(x.size(0), -1)
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)): feat = feat[0]
            if feat.dim() == 3: feat = feat.mean(dim=1)
            elif feat.dim() == 4: feat = feat.mean(dim=[2, 3])
            
        return self.heads[task](feat)

def evaluate_clinical(current_model, loader, task_name, num_classes):
    current_model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device).long()
            outs = current_model(imgs, task=task_name)
            probs = torch.softmax(outs, dim=1)
            _, preds = torch.max(outs, 1)
            
            all_preds.extend(preds.cpu().numpy()); all_targets.extend(lbls.cpu().numpy())
            if num_classes == 2: all_probs.extend(probs[:, 1].cpu().numpy())
            else: all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    try:
        if num_classes == 2: auc = roc_auc_score(all_targets, all_probs)
        else: auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except: auc = 0.5
    return {"acc": acc * 100, "auc": auc, "f1": f1}



if __name__ == '__main__':
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    Image.MAX_IMAGE_PIXELS = None



    task_loaders, task_classes = {}, {}

    # 1. LC2500
    p, l = [], []
    for r, _, fs in os.walk(DIRS["TCGA_LUNG"]):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.jpeg')):
                if 'lung_aca' in r.lower(): p.append(os.path.join(r, f)); l.append(0)
                elif 'lung_scc' in r.lower(): p.append(os.path.join(r, f)); l.append(1)
    if p: task_loaders["lung"], task_loaders["lung_val"], n = create_loader(p, l); task_classes["lung"] = 2

    # 2. BREAKHIS
    p, l = [], []
    for r, _, fs in os.walk(DIRS["BREAKHIS"]):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                if 'benign' in r.lower(): p.append(os.path.join(r, f)); l.append(0)
                elif 'malignant' in r.lower(): p.append(os.path.join(r, f)); l.append(1)
    if p: task_loaders["breakhis"], task_loaders["breakhis_val"], n = create_loader(p, l); task_classes["breakhis"] = 2


    p, l, num_c = scan_generic(DIRS["NCT_COLON"])
    if p: task_loaders["nct"], task_loaders["nct_val"], _ = create_loader(p, l); task_classes["nct"] = num_c

    p, l, num_c = scan_generic(DIRS["HUBMAP"])
    if p: task_loaders["hubmap"], task_loaders["hubmap_val"], _ = create_loader(p, l); task_classes["hubmap"] = num_c

    p, l, num_c = scan_generic(DIRS["BACH"])
    if p: task_loaders["bach"], task_loaders["bach_val"], _ = create_loader(p, l); task_classes["bach"] = num_c

    active_tasks = list(task_classes.keys())


    for MODEL_TO_RUN in MODELS_TO_RUN:


        SAVE_DIR = f"./risultati_finali_pancancer/{MODEL_TO_RUN}"
        os.makedirs(SAVE_DIR, exist_ok=True)


        if MODEL_TO_RUN == "resnet":
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            embed_dim = 2048

        elif MODEL_TO_RUN == "ctranspath":
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

                    backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, embed_dim=96)
                    backbone.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm)
                    backbone.head = nn.Identity()
                    
                    CTRANSPATH_WEIGHTS = "./weights/ctranspath.pth" 
                    
                    if os.path.exists(CTRANSPATH_WEIGHTS):
                        ckpt = torch.load(CTRANSPATH_WEIGHTS, map_location="cpu")
                        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
                        
                        model_dict = backbone.state_dict()
                        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                        
                        backbone.load_state_dict(pretrained_dict, strict=False)
                    else:
                        raise ValueError(f"Weights CTransPath not found in {CTRANSPATH_WEIGHTS}")
                    
                    # 3. Lo Swin-Tiny sputa fuori vettori di dimensione 768 alla fine
                    embed_dim = 768

       elif MODEL_TO_RUN == "phikon":
            from transformers import ViTModel
            base_vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            
            class PhikonWrapper(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.model = vit_model
                def forward(self, x):
                    return self.model(x, return_dict=False)[0] 
                    
            backbone = PhikonWrapper(base_vit)
            embed_dim = 768

        elif MODEL_TO_RUN == "uni":
            from timm.layers import SwiGLUPacked
            timm_kwargs = {
                'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 
                'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 
                'num_classes': 0, 'no_embed_class': True, 
                'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 'dynamic_img_size': True
            }
            backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            embed_dim = 1536

        elif MODEL_TO_RUN == "virchow2":
            backbone = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, num_classes=0, dynamic_img_size=True)
            embed_dim = 1280

        elif MODEL_TO_RUN == "conch":
                    import open_clip
                    from open_clip import factory
                    from huggingface_hub import hf_hub_download
                    
                    factory._MODEL_CONFIGS['conch_ViT-B-16'] = {
                        "embed_dim": 512, "vision_cfg": {"image_size": 224, "layers": 12, "width": 768, "patch_size": 16},
                        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12}
                    }
                    conch_weights_path = hf_hub_download(repo_id="MahmoodLab/conch", filename="pytorch_model.bin")
                    
                    full_model, _, _ = open_clip.create_model_and_transforms('conch_ViT-B-16', pretrained=None)
                    full_model.load_state_dict(torch.load(conch_weights_path, map_location="cpu"), strict=False)
                    backbone = full_model.visual
                    embed_dim = 512

        model = UniversalPanCancerModel(backbone, MODEL_TO_RUN, task_classes, embed_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

        max_batches = max([len(task_loaders[task]) for task in active_tasks])
        history = {task: [] for task in active_tasks}

        print(f"\nSTART TRAINING ({MODEL_TO_RUN.upper()}) ")

        for epoch in range(EPOCHS):
            start_time = time.time()
            model.train()
            iterators = {task: iter(task_loaders[task]) for task in active_tasks}
            
            for step in tqdm(range(max_batches), desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad() 
                
                for task in active_tasks:
                    try: img, lbl = next(iterators[task])
                    except StopIteration: 
                        iterators[task] = iter(task_loaders[task])
                        img, lbl = next(iterators[task])
                        
                    img, lbl = img.to(device), lbl.to(device).long()
                    loss = criterion(model(img, task=task), lbl)
                    loss.backward() 
                
                optimizer.step()

  
            print(f"\nEpoch {epoch+1} (Tempo: {time.time()-start_time:.0f}s)")
            for task in active_tasks:
                val_loader = task_loaders[f"{task}_val"]
                num_c = task_classes[task]
                metrics = evaluate_clinical(model, val_loader, task, num_c)
                history[task].append(metrics)
                print(f"➤ {task.upper():<10} | Acc: {metrics['acc']:.1f}% | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
            print("-" * 60)
            
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"PanCancer_{MODEL_TO_RUN}_Epoch_{epoch+1}.pth"))

        with open(os.path.join(SAVE_DIR, f"PanCancer_{MODEL_TO_RUN}_metrics.json"), "w") as f:
            json.dump(history, f, indent=4)

 

        del model
        del backbone
        del optimizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

