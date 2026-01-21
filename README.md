
# 🧬 Neuro-Symbolic Activation Discovery

[![Paper](https://img.shields.io/badge/arXiv-2026.2601-b31b1b.svg)](https://arxiv.org/abs/2601.10740)
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

## 📂 Project Structure

```text
NeuroSymbolic_Activation/
├── data/                  # Downloaded datasets (HIGGS.csv, etc.)
├── src/
│   ├── data_loader.py     # Dataset fetching and preprocessing
│   ├── models.py          # AutoSymbolicLayer, Heavy/Light Models
│   ├── discovery.py       # Genetic Programming logic (gplearn)
│   ├── train.py           # Training loop and evaluation metrics
│   └── utils.py           # Seeds, device, plotting helpers
├── results/               # Generated plots (activation_*.png) and CSV results
├── main.py                # Entry point: orchestrates the full pipeline
├── benchmark_standalone.py # Single-file script containing all logic
├── requirements.txt       # Python dependencies
└── README.md
```

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
```

---

## 🚀 Usage

### ⚡ Quick Start: Standalone Script

For users who want to run the benchmark immediately without managing the `src/` folder structure, we provide a **standalone script**. This single file (`benchmark_standalone.py`) contains all necessary logic (data loading, GP discovery, training, and evaluation).

1. Save the standalone code provided in the repo as `benchmark_standalone.py`.
2. Run it directly:

```bash
python benchmark_standalone.py
```

**Output:** This will download data, train models, and save plots/CSVs to the current directory.

---

### 🏗️ Modular Execution (Recommended for Research)

For researchers who wish to modify individual components (e.g., change the architecture in `models.py` or the GP function set in `discovery.py`), use the modular entry point.

To reproduce the full benchmark (Table 2 in the paper):

```bash
python main.py
```

This script will:
1. Download datasets automatically.
2. Discover activation formulas using Genetic Programming.
3. Train Heavy and Light models across 3 random seeds.
4. Save results to `results/final_efficiency_results.csv`.

### Output

Check the `results/` folder for:
*   `activation_HIGGS.png`: Visualization of the discovered physics formula.
*   `activation_FOREST_COVER.png`: Visualization of the ecology formula.
*   `activation_SPAMBASE.png`: Visualization of the spam formula.
*   `final_efficiency_results.csv`: Raw numbers for all experiments.

### Reproducibility

All experiments use fixed random seeds (42, 43, 44) for robustness. Ensure you are using Python 3.8+ to match package versions exactly.

---

## 🏗️ Citation

If you use this code or find our research helpful, please cite:

```bibtex
@article{,
  title={Neuro-Symbolic Activation Discovery: Transferring Mathematical Structures from Physics to Ecology for Parameter-Efficient Neural Networks},
  author={Hajbi, Anas},
  journal={arXiv preprint arXiv:2601.10740},
  year={2026}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [anas.hajbi@um6p.ma](mailto:anas.hajbi@um6p.ma).
