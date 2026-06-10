# Benchmarking Vision Foundation Models in Computational Pathology

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)


Official codebase for the paper **"Benchmarking Vision Foundation Models in Computational Pathology: A Multi-Task Benchmark on Public Datasets"**.

---

## Overview

Vision Foundation Models (VFMs) are rapidly replacing task-specific convolutional networks in computational pathology. Yet independent evidence across heterogeneous diagnostic tasks remains limited, particularly under resource-constrained clinical environments.

This repository provides a reproducible benchmark of **five representative pathology VFMs** under a standardized **frozen linear-probing pipeline**, evaluated across a clinically motivated taxonomy of **eight public datasets**:

- **Categorical Tissue Phenotyping:** BreaKHis, NCT-CRC-HE-100K, HuBMAP, BACH, LC25000
- **Ordinal Prognostic Grading:** PANDA, BRACS, SICAPv2

---

## Evaluated Models

| Model | Architecture | Parameters | Pre-training Data | SSL Method | Availability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UNI** | ViT-L/14 | 307M | 100K WSIs | DINOv2 | Gated |
| **Virchow2** | ViT-H/14 | 632M | 3.1M WSIs | DINOv2 | Gated |
| **CONCH** | ViT-B/16 | ~200M* | Reports + Patches | CLIP-style | Gated |
| **Phikon** | ViT-B/16 | 86M | TCGA | iBOT | Open |
| **CTransPath** | Swin-T | 28M | 32K WSIs | SRCL | Open |

> *The ~200M parameter count for CONCH includes both the ViT-B/16 visual backbone (86M) and the language encoder used during multimodal pre-training. Only the visual encoder is used as a frozen feature extractor during benchmarking.

---

## Repository Structure

```
FMs-CPathology/
├── comparison/                   # Single-task baseline notebooks (Table III)
│   ├── BACH_train_10_epochs.ipynb
│   ├── PANDA_train_10_epochs.ipynb
│   ├── break_10_epochs.ipynb     # BreaKHis
│   ├── hubmap_train_10_epochs.ipynb
│   ├── lc2500_10_epochs.ipynb
│   └── nct_train_10_epochs.ipynb
├── multiple/                     # Simultaneous multi-task pipelines (Tables IV–V)
│   ├── categorical/
│   │   └── multiple_categorical.py
│   └── ordinal_grading/
│       └── multiple_ordinal.py
├── quantization/                 # Hardware efficiency & reduced-precision inference (Table VI)
│   └── quantization.py
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/NelloC/FMs-CPathology.git
cd FMs-CPathology
```

Install dependencies (ensure your PyTorch build matches your CUDA version):

```bash
pip install -r requirements.txt
```

**Hardware used in the paper:** NVIDIA GeForce RTX 4090 (GPU benchmarks) and Apple Silicon M3 (CPU/INT8 benchmarks).

---

## Dataset Preparation

All datasets used in this benchmark are publicly available. Download them from their respective sources and place them under a `data/` directory:

| Dataset | Task | Source |
| :--- | :--- | :--- |
| NCT-CRC-HE-100K | Colorectal phenotyping | [Zenodo](https://zenodo.org/record/1214456) |
| HuBMAP | Multi-organ segmentation | [Kaggle](https://kaggle.com/competitions/hubmap-organ-segmentation) |
| BACH | Breast cancer classification | [ICIAR 2018](https://iciar2018-challenge.grand-challenge.org/) |
| BreaKHis | Breast tumor classification | [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) |
| LC25000 | Lung cancer classification | [arXiv](https://arxiv.org/abs/1912.12142) |
| PANDA | Prostate Gleason grading | [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment) |
| BRACS | Breast atypia subtyping | [Official site](https://www.bracs.icar.cnr.it/) |
| SICAPv2 | Prostate Gleason grading | [Official site](https://data.mendeley.com/datasets/9xxm58dvs3) |

All patches are extracted at **20× magnification**, resized to **224×224 pixels**, and normalized using standard ImageNet statistics.

> For PANDA, the benchmark uses a stratified 25% subsample for multi-task evaluation (Table V) to maintain scale parity with other datasets. The single-task baseline (Table III) uses the full cohort.

---

## Usage

### 1. Single-Task Baselines (Table III)

Individual Jupyter notebooks under `comparison/` reproduce the single-task frozen linear-probing results for each dataset:

```bash
jupyter notebook comparison/BACH_train_10_epochs.ipynb
```

### 2. Simultaneous Multi-Task Learning (Tables IV–V)

To evaluate a frozen backbone across multiple tasks concurrently (categorical or ordinal):

```bash
# Categorical benchmark (Table IV)
python multiple/categorical/multiple_categorical.py --model uni

# Ordinal grading benchmark (Table V)
python multiple/ordinal_grading/multiple_ordinal.py --model virchow2
```

Available `--model` options: `uni`, `virchow2`, `conch`, `phikon`, `ctranspath`

### 3. Quantization Benchmark (Table VI)

To reproduce the FP32 / FP16 / INT8 precision comparison:

```bash
python quantization/quantization.py --model phikon
```

---

## Key Findings

- **Categorical tasks:** Massive unimodal models (UNI, Virchow2) maintain macro-average AUC > 0.99 across five distinct tissue types under simultaneous multi-task training, demonstrating strong resistance to catastrophic forgetting.
- **Ordinal grading:** Frozen patch-level representations collapse on WSI-derived grading tasks (PANDA) due to global-to-local label noise. Clean RoI datasets (BRACS, SICAPv2) do not exhibit this failure. Addressing PANDA-style label noise requires spatial aggregation frameworks (e.g., ABMIL, CLAM) rather than linear probing.
- **Positive transfer:** Co-training on related breast tissue tasks (BACH + BreaKHis) yields a +8.84% accuracy gain for UNI on BACH compared to single-task isolation.
- **Quantization resilience:** High-parameter unimodal models tolerate INT8 CPU quantization with negligible accuracy degradation on categorical tasks, making them viable for resource-constrained clinical deployment.

