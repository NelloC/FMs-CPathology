# Benchmarking Vision Foundation Models in Computational Pathology: A Multi-Task Benchmark on Public Datasets

This repository contains the official code and evaluation benchmark for the paper **"Benchmarking Vision Foundation Models in Computational Pathology: A Multi-Task Benchmark on Public Datasets"** 

## Overview

Vision Foundation Models (VFMs) are rapidly transforming computational pathology by replacing task-specific convolutional networks with generalist backbones trained via self-supervised learning[cite: 1]. However, independent evidence evaluating these models across heterogeneous diagnostic tasks remains limited[cite: 1]. 

This project benchmarks five representative pathology VFMs to quantify the trade-offs between model scale, downstream diagnostic performance, and computational efficiency[cite: 1]. We evaluate these models across a novel, clinically motivated taxonomy covering eight public datasets[cite: 1]:
*   **Categorical Tissue Phenotyping:** BACH, BreaKHis, NCT-CRC, HUBMAP, LC25000[cite: 1].
*   **Ordinal Prognostic Grading:** PANDA, BRACS, SICAPv2[cite: 1].

## Evaluated Foundation Models

The benchmark evaluates the frozen embeddings of the following architectures using a standardized linear-probing pipeline[cite: 1]:

| Model | Architecture | Parameters | Pre-training Data | Availability |
| :--- | :--- | :--- | :--- | :--- |
| **UNI**[cite: 1] | ViT-L/14[cite: 1] | 307M[cite: 1] | 100K WSIs[cite: 1] | Gated[cite: 1] |
| **Virchow2**[cite: 1] | ViT-H/14[cite: 1] | 632M[cite: 1] | 3.1M WSIs[cite: 1] | Gated[cite: 1] |
| **CONCH**[cite: 1] | ViT-B/16[cite: 1] | ~200M[cite: 1] | Reports + Patches[cite: 1] | Gated[cite: 1] |
| **Phikon**[cite: 1] | ViT-B/16[cite: 1] | 86M[cite: 1] | TCGA[cite: 1] | Open[cite: 1] |
| **CTransPath**[cite: 1] | Swin-T[cite: 1] | 28M[cite: 1] | 32K WSIs[cite: 1] | Open[cite: 1] |

## Repository Structure

The repository is logically divided into independent modules based on the evaluation pipelines. Scripts can be executed in any order[cite: 1]:

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
