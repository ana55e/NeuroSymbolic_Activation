import torch
import torch.nn as nn
import re
import torchinfo

class AutoSymbolicLayer(nn.Module):
    """
    Compiles a string formula into a differentiable PyTorch layer.
    Replaces specific features (X1, X2) with generic input 'x'.
    """
    def __init__(self, formula_str):
        super().__init__()
        self.ops = {
            'add': torch.add, 'sub': torch.sub, 'mul': torch.mul, 'div': torch.div,
            'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan,
            'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6),
            'log': lambda x: torch.where(torch.abs(x) > 1e-6, torch.log(torch.abs(x)), torch.tensor(0.0).to(x.device)),
            'abs': torch.abs, 'neg': torch.neg, 'max': torch.maximum, 'min': torch.minimum, 
            'torch': torch
        }
        
        # Generalize: Replace specific features (X1, X2) with generic 'x'
        clean = re.sub(r'[X|x]\d+', 'x', formula_str)
        
        # Compile to Python function
        for op in self.ops.keys():
            if op != 'torch':
                clean = re.sub(rf'\b{op}\(', f'ops["{op}"](', clean)
        
        scope = {}
        try:
            exec(f"def dynamic_forward(x, ops):\n    return {clean}", scope)
        except Exception as e:
            print(f"⚠️ Formula Error ({e}). Fallback to Identity.")
            exec(f"def dynamic_forward(x, ops):\n    return x", scope)
        self.compiled_func = scope['dynamic_forward']

    def forward(self, x):
        return self.compiled_func(x, self.ops)

class HeavyModel(nn.Module):
    """
    The Industry Standard Baseline.
    Large, over-parameterized, uses ReLU.
    Params: ~26,000
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(200, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class LightModel(nn.Module):
    """
    The Efficiency Target.
    Small, compact. 
    Can use: ReLU, GELU, SiLU, or HYBRID (Discovered).
    Params: ~4,000 (~6x smaller than Heavy)
    """
    def __init__(self, input_dim, act_type, formula=None):
        super().__init__()
        
        # Select Activation
        if act_type == "Hybrid":
            act = AutoSymbolicLayer(formula)
        elif act_type == "GELU":
            act = nn.GELU()
        elif act_type == "SiLU":
            act = nn.SiLU() 
        else: # Default to ReLU
            act = nn.ReLU()
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), act, nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), act, nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
