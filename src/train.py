import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
import torchinfo

from .utils import DEVICE, set_seed
from .models import HeavyModel, LightModel  

def run_experiment(X, y, dataset_name, model_class, act_type, formula=None):
    """
    Runs a single experiment configuration 3 times (robustness).
    Returns a dictionary of results.
    """
    accs, aucs, effs = [], [], []
    params_count = 0
    SEEDS = [42, 43, 44]
    
    for seed in SEEDS:
        set_seed(seed)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Scale
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
        ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(ds, batch_size=1024, shuffle=True)
        
        model.train()
        for epoch in range(15): 
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
