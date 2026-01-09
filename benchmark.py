import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype, fetch_openml
from gplearn.genetic import SymbolicClassifier
import mlflow
import requests
import gzip
import shutil
import os
import re
import random
import torchinfo

# ==========================================
# 1. SETUP
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_experiment("Final_Benchmark_With_Efficiency")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print(f"🚀 Running on: {DEVICE}")

# ==========================================
# 2. DATA LOADER
# ==========================================
def get_dataset(name, limit_rows=100000):
    print(f"\n[Data] Fetching {name}...")
    if name == "HIGGS":
        if not os.path.exists("HIGGS.csv"):
            print("  Downloading HIGGS...")
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
# 3. AUTO SYMBOLIC LAYER
# ==========================================
class AutoSymbolicLayer(nn.Module):
    def __init__(self, formula_str, mode='general'):
        super().__init__()
        self.mode = mode
        self.ops = {
            'add': torch.add, 'sub': torch.sub, 'mul': torch.mul, 'div': torch.div,
            'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan,
            'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6),
            'log': lambda x: torch.where(torch.abs(x) > 1e-6, torch.log(torch.abs(x)), torch.tensor(0.0).to(x.device)),
            'abs': torch.abs, 'neg': torch.neg, 'torch': torch
        }
        self.compiled_func = self._compile(formula_str)

    def _compile(self, formula):
        clean = formula
        for op in ['sin', 'cos', 'tan', 'sqrt', 'log', 'abs', 'neg', 'add', 'sub', 'mul', 'div']:
            clean = re.sub(rf'\b{op}\(', f'ops["{op}"](', clean)
        
        if self.mode == 'precise':
            clean = re.sub(r'\b[xX](\d+)\b', r'x[:, \1]', clean)
        elif self.mode == 'general':
            clean = re.sub(r'\b[xX]\d+\b', r'x', clean)
            
        scope = {}
        try:
            exec(f"def dynamic_forward(x, ops):\n    return {clean}", scope)
        except:
            exec(f"def dynamic_forward(x, ops):\n    return x * 0 + {clean}", scope)
        return scope['dynamic_forward']

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        res = self.compiled_func(x, self.ops)
        if res.dim() == 1: res = res.unsqueeze(1)
        return res

# ==========================================
# 4. MODELS
# ==========================================
class HeavyANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(200, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class LightANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class HybridSpecialist(nn.Module):
    def __init__(self, input_dim, formula):
        super().__init__()
        self.act = AutoSymbolicLayer(formula, mode='general')
        self.feat = AutoSymbolicLayer(formula, mode='precise')
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.head = nn.Linear(33, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.bn1(self.fc1(x))
        out = self.act(out)
        out = self.bn2(self.fc2(out))
        out = self.act(out)
        shortcut = self.feat(x)
        return self.sigmoid(self.head(torch.cat([out, shortcut], dim=1)))

class HybridTransfer(nn.Module):
    def __init__(self, input_dim, formula):
        super().__init__()
        self.act = AutoSymbolicLayer(formula, mode='general')
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.bn1(self.fc1(x))
        out = self.act(out)
        out = self.bn2(self.fc2(out))
        out = self.act(out)
        return self.sigmoid(self.head(out))

# ==========================================
# 5. TRAINING HELPER (UPDATED WITH EFFICIENCY)
# ==========================================
def train_and_eval(model, train_loader, X_test, y_test, run_name, params):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()
    
    # 1. Calculate Parameters
    dummy_input = torch.zeros(1, X_test.shape[1]).to(DEVICE)
    model_stats = torchinfo.summary(model, input_data=dummy_input, verbose=0)
    total_params = model_stats.total_params
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("total_params", total_params)
        
        # Train
        model.train()
        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
            probs = model(x_t)
            test_loss = crit(probs, y_t.unsqueeze(1)).item()
            probs = probs.cpu().numpy()
            
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        # 2. Calculate Efficiency Score (Metric / log10(Params))
        # Higher is better. We use AUC as the numerator.
        eff_score = auc / np.log10(total_params)
        
        print(f"   -> Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Loss: {test_loss:.4f} | Eff: {eff_score:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("efficiency_score", eff_score)

# ==========================================
# 6. EXECUTION
# ==========================================
datasets = ["HIGGS", "FOREST_COVER", "SPAMBASE"]
higgs_formula = None

print("\n" + "="*50)
print("🧪 CASE 1: SPECIALIST")
print("="*50)

for name in datasets:
    print(f"\n--- Processing {name} ---")
    X, y = get_dataset(name)
    
    X_disc, X_rest, y_disc, y_rest = train_test_split(X, y, test_size=0.9, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_rest, y_rest, test_size=0.2, random_state=42)
    
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_disc = s.transform(X_disc)
    X_test = s.transform(X_test)
    
    print("   Discovering Physics Formula...")
    gp = SymbolicClassifier(generations=5, population_size=500, 
                           function_set=['add', 'sub', 'mul', 'sin', 'cos', 'abs'],
                           random_state=42, n_jobs=-1)
    gp.fit(X_disc, y_disc)
    formula = str(gp._program)
    print(f"   Found: {formula}")
    
    if name == "HIGGS":
        higgs_formula = formula

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                       torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=1024, shuffle=True)

    models = [
        ("Heavy_ANN", HeavyANN(X.shape[1])),
        ("Light_ANN", LightANN(X.shape[1])),
        ("Hybrid_Specialist", HybridSpecialist(X.shape[1], formula))
    ]
    
    for m_name, model in models:
        print(f"   Training {m_name}...")
        train_and_eval(model, loader, X_test, y_test, 
                       run_name=f"Case1_{name}_{m_name}",
                       params={"dataset": name, "model": m_name, "case": "Specialist"})

print("\n" + "="*50)
print("🚀 CASE 2: TRANSFER")
print("="*50)

for name in ["FOREST_COVER", "SPAMBASE"]:
    print(f"\n--- Transfer to {name} ---")
    X, y = get_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)
    
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                       torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=1024, shuffle=True)
    
    models = [
        ("Light_ANN_Control", LightANN(X.shape[1])),
        ("Hybrid_Transfer", HybridTransfer(X.shape[1], higgs_formula))
    ]
    
    for m_name, model in models:
        print(f"   Training {m_name}...")
        train_and_eval(model, loader, X_test, y_test, 
                       run_name=f"Case2_{name}_{m_name}",
                       params={"dataset": name, "model": m_name, "case": "Transfer"})

print("\n✅ BENCHMARK COMPLETE.")
