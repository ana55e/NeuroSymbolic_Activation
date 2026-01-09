# Bio-Physical Activation Discovery: Neuro-Symbolic Efficiency for Science

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper **"Bio-Physical Activation Discovery: Transferring Mathematical Structures to Enhance Neural Efficiency."**

We introduce a Neuro-Symbolic pipeline that uses **Genetic Programming (GP)** to discover mathematical formulas from data and inject them as custom **Activation Functions** into Neural Networks. This approach bridges the gap between interpretable symbolic regression and deep learning, yielding models that are **6x smaller** than standard baselines while maintaining high accuracy.

## 🚀 Key Findings

1.  **Efficiency Win:** Our Hybrid models achieve **20-23% higher Efficiency Scores** ($E = \mathrm{AUC} / \log_{10}(\mathrm{Params})$) compared to standard "Heavy" ANNs.
2.  **Physics Transfer:** We demonstrate that activation functions learned from Particle Physics (HIGGS) transfer successfully to Ecological tasks (Forest Cover), suggesting a shared "Geometric Grammar" in continuous scientific domains.
3.  **Low-Compute:** Discovery requires only CPUs and 10% of the training data.

## 📂 Repository Structure

- `benchmark.py`: The main script. Runs the full experiment suite (Specialist Discovery + Transfer Learning) across 3 datasets.
- `README.md`: This file.

## 🛠️ Installation

This project requires **PyTorch**, **scikit-learn**, **gplearn**, and **MLflow**.

```bash
pip install torch torchvision numpy pandas scikit-learn gplearn mlflow torchinfo
