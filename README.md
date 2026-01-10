# 🧬 Neuro-Symbolic Activation Discovery

[![Paper](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

> **Transferring Mathematical Structures from Physics to Ecology for Parameter-Efficient Neural Networks**

This repository contains the official implementation of our paper on discovering domain-specific activation functions using Genetic Programming and transferring them across scientific domains.

---

## 📋 Abstract

Modern neural networks rely on generic activation functions (ReLU, GELU, SiLU) that ignore the mathematical structure inherent in scientific data. We propose **Neuro-Symbolic Activation Discovery**, a framework that uses Genetic Programming to extract interpretable mathematical formulas from data and inject them as custom activation functions.

**Key Findings:**
- 🎯 **Geometric Transfer**: Activation functions discovered on particle physics data successfully generalize to ecological classification
- ⚡ **Efficiency**: 18-21% higher parameter efficiency with 5-6× fewer parameters
- 🔬 **Interpretability**: Human-readable symbolic formulas as activation functions

---

## 🚀 Key Results

| Dataset | Best Model | Accuracy | Params | Efficiency Gain |
|---------|------------|----------|--------|-----------------|
| HIGGS | Light ReLU | 71.0% | 4,161 | +21.2% vs Heavy |
| Forest Cover | **Hybrid (Transfer)** | **82.4%** | 5,825 | +18.2% vs Heavy |
| Spambase | **Hybrid (Specialist)** | **92.0%** | 6,017 | +18.0% vs Heavy |

**The Transfer Phenomenon**: A formula discovered on HIGGS (`mul(cos(x), x)`) transfers to Forest Cover, outperforming ReLU, GELU, and SiLU!

---

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/ana55e/NeuroSymbolic_Activation.git
cd NeuroSymbolic_Activation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
