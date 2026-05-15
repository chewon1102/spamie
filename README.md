<img width="595" height="294" alt="image" src="https://github.com/user-attachments/assets/12f4c2c4-82e1-400f-81cc-fd2ae2e267ca" /># SpaMIE: Spatial Multi-Omics Imputation and Embedding

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue)](https://www.sciencedirect.com/science/article/pii/S2667237526001566)
[![Documentation Status](https://img.shields.io/badge/docs-tutorials-brightgreen.svg)](https://spamie-tutorials.readthedocs.io/en/latest/)

**SpaMIE** is a deep graph neural network framework designed to tackle the challenge of multi-section integration in spatial multi-omics (SMO) datasets with systematically missing modalities. 

While spatially resolved molecular profiling provides profound insights into tissue biology, the high cost and technical complexity of joint multi-omics profiling within a single section have limited its widespread use. Consequently, large-scale spatial atlases predominantly rely on cost-effective mono-omics measurements, resulting in heterogeneous modality coverage across tissue sections. Existing methods often ignore spatial dependencies or assume all modalities are jointly observed. 

SpaMIE addresses this analytical challenge by offering a robust **two-stage solution**:
1. **Spatially Informed Cross-Modal Imputation**: Accurately infers missing modalities from mono-omics data by fully leveraging spatial dependencies.
2. **Joint Embedding and Integration**: Integrates both measured and imputed spatial multi-omics profiles across multiple tissue sections to learn a unified embedding.

Check out our [published paper](https://www.sciencedirect.com/science/article/pii/S2667237526001566) and our [Tutorial Website](https://spamie-tutorials.readthedocs.io/en/latest/) for a complete description of the methodology and downstream analyses.

## Key Highlights
* **Cross-Modal Imputation**: Enables spatially informed cross-modal imputation specifically tailored for systematically missing modalities.
* **Stabilized Two-Stage Design**: Effectively stabilizes both imputation and integration across complex tissue sections.
* **Unified Embeddings**: Learns cohesive and aligned embeddings across sections with highly heterogeneous modality coverage.
* **Scalable Framework**: Provides a flexible foundation for building and analyzing large-scale spatial multi-omics atlases, demonstrating robust performance in downstream tasks such as spatial domain identification.

## Overview
![](https://github.com/xxdwdwd/SpaMIE/blob/main/overview.png)

## Installation

### 1. Create and activate a conda environment
```bash
git clone [https://github.com/xxdwdwd/SpaMIE.git](https://github.com/xxdwdwd/SpaMIE.git)
cd SpaMIE
conda create -n SpaMIE_env python=3.9
conda activate SpaMIE_env

### 2. Install core dependencies
```bash
# dependencies
pip install -r requirements.txt
# SpaMIE
pip install -e .
# or pip install dist/SpaMIE-0.1.0-py3-none-any.whl
```
