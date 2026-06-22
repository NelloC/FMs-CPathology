import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import timm
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import login
import types
import time
import json

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

MODELS_TO_RUN = ["ctranspath", "phikon", "uni", "virchow2", "conch"]

BATCH_SIZE = 4 
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 6

TILES_DIR = "/home/nelloconelli/Escritorio/FOUNDATION MODELS/PANDA/DATASET/PANDA_TILED_TIFF"
CSV_PATH = "/home/nelloconelli/Escritorio/FOUNDATION MODELS/PANDA/DATASET/PANDA/train.csv"
SAVE_DIR = "./risultati_finali"
CTRANSPATH_WEIGHTS = "./model_lib/pretrained/ctranspath.pth"

Image.MAX_IMAGE_PIXELS = None
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(CSV_PATH)
patient_ids = df['image_id'].values

train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42, stratify=df['isup_grade'])

class PandaTileDataset(Dataset):
    def __init__(self, root_dir, patient_ids_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        valid_ids = set(patient_ids_list)
        
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path): continue
            
            for img_name in os.listdir(class_path):
                p_id = img_name.split('_')[0]
                if p_id in valid_ids:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(int(class_folder)) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros(3, 224, 224), 0

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = PandaTileDataset(TILES_DIR, train_ids, transform=train_transform)
val_dataset = PandaTileDataset(TILES_DIR, val_ids, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

if len(train_dataset.labels) > 0:
    cw = compute_class_weight('balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

for CURRENT_MODEL in MODELS_TO_RUN:
    if CURRENT_MODEL == "ctranspath":
        def to_2tuple(x): return tuple(x) if isinstance(x, (tuple, list)) else (x, x)
        class ConvStem(nn.Module):
            def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim // 2), nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
                )
            def forward(self, x): return self.stem(x).permute(0, 2, 3, 1)

        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, embed_dim=96)
        model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm)
        
        if os.path.exists(CTRANSPATH_WEIGHTS):
            ckpt = torch.load(CTRANSPATH_WEIGHTS, map_location="cpu")
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            model_dict = model.state_dict()
            valid_weights = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model.load_state_dict(valid_weights, strict=False)

        model.head = nn.Linear(768, NUM_CLASSES)
        def pooling_forward(self, x):
            x = self.forward_features(x)
            x = x.mean(dim=[1, 2]) 
            x = self.head(x)
            return x
        model.forward = types.MethodType(pooling_forward, model)
        
        for param in model.parameters(): param.requires_grad = False
        for param in model.head.parameters(): param.requires_grad = True

    elif CURRENT_MODEL == "phikon":
        from transformers import ViTModel
        base_vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        class PhikonClassifier(nn.Module):
            def __init__(self, vit_model, num_classes):
                super().__init__()
                self.backbone = vit_model
                for param in self.backbone.parameters(): param.requires_grad = False
                self.head = nn.Linear(768, num_classes)
            def forward(self, x):
                feat = self.backbone(x, return_dict=False)[0]
                return self.head(feat[:, 0, :])
        model = PhikonClassifier(base_vit, NUM_CLASSES)

    elif CURRENT_MODEL == "uni":
        from timm.layers import SwiGLUPacked
        timm_kwargs = {
            'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 
            'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 
            'num_classes': 0, 'no_embed_class': True, 
            'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 'dynamic_img_size': True
        }
        backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        class UNIWrapper(nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                for param in self.backbone.parameters(): param.requires_grad = False
                self.head = nn.Linear(1536, num_classes)
            def forward(self, x): return self.head(self.backbone(x))
        model = UNIWrapper(backbone, NUM_CLASSES)

    elif CURRENT_MODEL == "virchow2":
        class Virchow2Classifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = timm.create_model(
                    "hf-hub:paige-ai/Virchow2", pretrained=True, 
                    mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                    num_classes=0, dynamic_img_size=True     
                )
                for param in self.backbone.parameters(): param.requires_grad = False
                self.head = nn.Linear(1280, num_classes)
            def forward(self, x):
                features = self.backbone(x)
                if isinstance(features, (list, tuple)): features = features[0]
                if features.dim() == 3: features = features.mean(dim=1)
                elif features.dim() == 4: features = features.mean(dim=[2, 3])
                return self.head(features)
        model = Virchow2Classifier(NUM_CLASSES)

    elif CURRENT_MODEL == "conch":
        import open_clip
        from open_clip import factory 
        from huggingface_hub import hf_hub_download

        factory._MODEL_CONFIGS['conch_ViT-B-16'] = {
            "embed_dim": 512,
            "vision_cfg": {"image_size": 224, "layers": 12, "width": 768, "patch_size": 16},
            "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12}
        }
        class CONCHClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                base_model, _, _ = open_clip.create_model_and_transforms('conch_ViT-B-16', pretrained=None)
                checkpoint_path = hf_hub_download(repo_id="MahmoodLab/CONCH", filename="pytorch_model.bin")
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
                base_model.load_state_dict(ckpt, strict=False)
                self.backbone = base_model.visual
                for param in self.backbone.parameters(): param.requires_grad = False
                self.head = nn.Linear(512, num_classes)
            def forward(self, x):
                features = self.backbone(x)
                if isinstance(features, tuple): features = features[0]
                return self.head(features)
        model = CONCHClassifier(NUM_CLASSES)

    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    history = {'epochs': [], 'loss': [], 'val_acc': [], 'time': []} 
    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time() 

        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_time = time.time() - start_time 
        val_acc = 100 * val_correct / val_total
        epoch_loss = running_loss / len(train_loader)
        
        history['epochs'].append(epoch+1)
        history['loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        history['time'].append(epoch_time)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{CURRENT_MODEL}_best.pth"))

    json_path = os.path.join(SAVE_DIR, f"stats_{CURRENT_MODEL}.json")
    with open(json_path, "w") as f:
        json.dump(history, f)

    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
