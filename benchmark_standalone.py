import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype, fetch_openml
from gplearn.genetic import SymbolicClassifier
import requests
import gzip
import shutil
import os
import re
import random
import torchinfo
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {DEVICE}")

# Robustness: We run every configuration 3 times
SEEDS = [42, 43, 44] 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. DATA LOADING
# ==========================================
def get_dataset(name, limit_rows=100000):
    print(f"\n[Data] Fetching {name}...")
    if name == "HIGGS":
        if not os.path.exists("HIGGS.csv"):
            print("  Downloading HIGGS (This might take a while)...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            with requests.get(url, stream=True) as r:
                with open("HIGGS.csv.gz", 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with gzip.open("HIGGS.csv.gz", 'rb') as f_in:
                with open("HIGGS.csv", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        df = pd.read_csv("HIGGS.csv", header=None, nrows=limit_rows)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    elif name == "FOREST_COVER":
        data = fetch_covtype()
        X = data.data
        y = (data.target == 2).astype(int) 
        if len(y) > limit_rows:
            idx = np.random.choice(len(y), limit_rows, replace=False)
            X, y = X[idx], y[idx]
    elif name == "SPAMBASE":
        data = fetch_openml(name='spambase', version=1, as_frame=False)
        X = data.data
        y = data.target.astype(int)
    return X, y

# ==========================================
# 3. NEURO-SYMBOLIC ACTIVATION
# ==========================================
class AutoSymbolicLayer(nn.Module):
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
        except:
            # Fallback if formula is invalid
            print(f"⚠️ Formula Error. Fallback to Identity.")
            exec(f"def dynamic_forward(x, ops):\n    return x", scope)
        self.compiled_func = scope['dynamic_forward']

    def forward(self, x):
        return self.compiled_func(x, self.ops)

def visualize_activation(formula, name):
    """Generates the figure for the paper."""
    try:
        x = torch.linspace(-5, 5, 100)
        relu = torch.relu(x)
        swish = torch.nn.functional.silu(x)
        
        # Create a dummy model just to run the formula
        layer = AutoSymbolicLayer(formula)
        y_ours = layer(x.unsqueeze(1)).squeeze()
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, relu, label='ReLU', linestyle='--', color='gray', alpha=0.5)
        plt.plot(x, swish, label='Swish', linestyle=':', color='green', alpha=0.5)
        plt.plot(x, y_ours, label='Discovered', linewidth=2, color='red')
        plt.title(f"Activation: {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"activation_{name}.png")
        plt.close()
        print(f"  [Plot] Saved activation_{name}.png")
    except Exception as e:
        print(f"  [Plot Error] Could not plot formula: {e}")

# ==========================================
# 4. MODEL ARCHITECTURES (HEAVY vs LIGHT)
# ==========================================
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
            act = nn.SiLU() # Swish
        else: # Default to ReLU
            act = nn.ReLU()
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), act, nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), act, nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def run_experiment(X, y, dataset_name, model_class, act_type, formula=None):
    accs, aucs, effs = [], [], []
    params_count = 0
    
    # Run 3 times for robustness
    for seed in SEEDS:
        set_seed(seed)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)
        
        # Init Model
        if model_class == "Heavy":
            model = HeavyModel(X.shape[1]).to(DEVICE)
        else:
            model = LightModel(X.shape[1], act_type, formula).to(DEVICE)
            
        # Count Params (only once)
        if seed == SEEDS[0]:
            dummy = torch.zeros(1, X.shape[1]).to(DEVICE)
            params_count = torchinfo.summary(model, input_data=dummy, verbose=0).total_params
        
        # Train
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()
        ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                           torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(ds, batch_size=1024, shuffle=True)
        
        model.train()
        for epoch in range(15): # 15 Epochs
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            probs = model(xt).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        # Efficiency Score: AUC per Log10(Param)
        # Why log? Because params vary by orders of magnitude (4k vs 26k).
        eff = auc / np.log10(params_count)
        
        accs.append(acc)
        aucs.append(auc)
        effs.append(eff)

    return {
        "Dataset": dataset_name,
        "Architecture": model_class,
        "Activation": act_type if model_class == "Light" else "ReLU",
        "Acc": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        "AUC": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        "Params": params_count,
        "Efficiency": f"{np.mean(effs):.3f}"
    }

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
final_results = []
datasets = ["HIGGS", "FOREST_COVER", "SPAMBASE"]
higgs_formula = None

print("\n" + "="*60)
print("🚀 NEURO-SYMBOLIC EFFICIENCY BENCHMARK")
print("="*60)

for name in datasets:
    print(f"\n--- Processing {name} ---")
    X, y = get_dataset(name)
    
    # 1. DISCOVERY PHASE (On 10% subset)
    set_seed(42)
    idx_disc = np.random.choice(len(X), int(0.1 * len(X)), replace=False)
    X_disc, y_disc = X[idx_disc], y[idx_disc]
    s_disc = StandardScaler()
    X_disc = s_disc.fit_transform(X_disc)
    
    print(f"  [Phase 1] Genetic Discovery running...")
    gp = SymbolicClassifier(generations=5, population_size=500, 
                           function_set=['add', 'sub', 'mul', 'sin', 'cos', 'abs'],
                           random_state=42, n_jobs=-1)
    gp.fit(X_disc, y_disc)
    formula = str(gp._program)
    print(f"  > Discovered: {formula}")
    visualize_activation(formula, name)
    
    # Save HIGGS formula for transfer
    if name == "HIGGS": higgs_formula = formula
    
    # 2. TRAINING PHASE (Comparisons)
    print(f"  [Phase 2] Training Models (3 Seeds each)...")
    
    # A. The Heavy Baseline (The thing we want to beat in efficiency)
    res = run_experiment(X, y, name, "Heavy", "ReLU")
    final_results.append(res)
    print(f"    Heavy (ReLU): Eff={res['Efficiency']} | AUC={res['AUC']}")
    
    # B. The Light Baselines (Standard activations)
    for act in ["ReLU", "GELU", "SiLU"]:
        res = run_experiment(X, y, name, "Light", act)
        final_results.append(res)
        print(f"    Light ({act}): Eff={res['Efficiency']} | AUC={res['AUC']}")
        
    # C. The Hybrid Model (Specialist)
    res = run_experiment(X, y, name, "Light", "Hybrid", formula)
    res["Activation"] = "Hybrid (Specialist)"
    final_results.append(res)
    print(f"    Hybrid (Specialist): Eff={res['Efficiency']} | AUC={res['AUC']}")
    
    # D. The Hybrid Model (Transfer) - ONLY for Forest/Spam
    if name != "HIGGS" and higgs_formula:
        res = run_experiment(X, y, name, "Light", "Hybrid", higgs_formula)
        res["Activation"] = "Hybrid (Transfer)"
        final_results.append(res)
        print(f"    Hybrid (Transfer): Eff={res['Efficiency']} | AUC={res['AUC']}")

# ==========================================
# 7. RESULTS TABLE
# ==========================================
print("\n" + "="*60)
print("🏆 FINAL BENCHMARK RESULTS")
print("="*60)
df = pd.DataFrame(final_results)
# Sort by Dataset then Efficiency
df = df.sort_values(by=["Dataset", "Efficiency"], ascending=[True, False])
print(df.to_markdown(index=False))
df.to_csv("final_efficiency_results.csv", index=False)
