# Benchmarking Vision Foundation Models in Computational Pathology: A Multi-Task Benchmark on Public Datasets

This repository contains the official code and evaluation benchmark for the paper **"Benchmarking Vision Foundation Models in Computational Pathology: A Multi-Task Benchmark on Public Datasets"** 

## Overview

Vision Foundation Models (VFMs) are rapidly transforming computational pathology by replacing task-specific convolutional networks with generalist backbones trained via self-supervised learning. However, independent evidence evaluating these models across heterogeneous diagnostic tasks remains limited. 

This project benchmarks five representative pathology VFMs to quantify the trade-offs between model scale, downstream diagnostic performance, and computational efficiency. We evaluate these models across a novel, clinically motivated taxonomy covering eight public datasets:
*   **Categorical Tissue Phenotyping:** BACH, BreaKHis, NCT-CRC, HUBMAP, LC25000.
*   **Ordinal Prognostic Grading:** PANDA, BRACS, SICAPv2.

## Evaluated Foundation Models

The benchmark evaluates the frozen embeddings of the following architectures using a standardized linear-probing pipeline:

| Model | Architecture | Parameters | Pre-training Data | Availability |
| :--- | :--- | :--- | :--- | :--- |
| **UNI** | ViT-L/14 | 307M | 100K WSIs| Gated |
| **Virchow2** | ViT-H/14 | 632M | 3.1M WSIs | Gated|
| **CONCH**| ViT-B/16 | ~200M | Reports + Patches | Gated |
| **Phikon** | ViT-B/16 | 86M| TCGA | Open |
| **CTransPath** | Swin-T | 28M | 32K WSIs| Open |

## Repository Structure

The repository is logically divided into independent modules based on the evaluation pipelines. Scripts can be executed in any order:

*   `comparison/`
    *   `BACH_train_10_epochs.ipynb`
    *   `PANDA_train_10_epochs.ipynb`
    *   `break_10_epochs.ipynb`
    *   `hubmap_train_10_epochs.ipynb`
    *   `lc2500_10_epochs.ipynb`
    *   `nct_train_10_epochs.ipynb`
*   `multiple/`
    *   `categorical/`
        *   `multiple_categorical.py`
    *   `ordinal_grading/`
        *   `multiple_ordinal.py`
*   `quantization/`
    *   `quantization.py`
*   `.gitignore`

## Installation

To set up the environment and run the benchmark, we recommend using a Python virtual environment or Conda.

1. Clone the repository:
```bash
   git clone [https://github.com/NelloC/FMs-CPathology.git](https://github.com/NelloC/FMs-CPathology.git)
   cd FMs-CPathology
