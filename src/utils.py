import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ==========================================
# DEVICE SETUP
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Default Device: {DEVICE}")

def set_seed(seed):
    """Sets the seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def visualize_activation(formula, name, save_dir="results"):
    """Generates the activation function plot for the paper."""
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        
        from .models import AutoSymbolicLayer
        
        x = torch.linspace(-5, 5, 100)
        relu = torch.relu(x)
        swish = torch.nn.functional.silu(x)
        
        # Create layer and compute output
        layer = AutoSymbolicLayer(formula)
        with torch.no_grad():
            y_ours = layer(x.unsqueeze(1)).squeeze()
        
        plt.figure(figsize=(8, 5))
        plt.plot(x.numpy(), relu.numpy(), label='ReLU', linestyle='--', color='gray', alpha=0.5)
        plt.plot(x.numpy(), swish.numpy(), label='Swish', linestyle=':', color='green', alpha=0.5)
        plt.plot(x.numpy(), y_ours.numpy(), label='Discovered', linewidth=2, color='red')
        plt.title(f"Activation: {name}")
        plt.xlabel("Input x")
        plt.ylabel("Output f(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/activation_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Saved {save_dir}/activation_{name}.png")
        
    except Exception as e:
        print(f"  [Plot Error] Could not plot formula: {e}")
