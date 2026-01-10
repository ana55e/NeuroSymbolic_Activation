import pandas as pd
import numpy as np
import os

# Import local modules
from src.utils import DEVICE, visualize_activation
from src.data_loader import get_dataset
from src.train import run_experiment
from src.discovery import discover_formula

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    final_results = []
    datasets = ["HIGGS", "FOREST_COVER", "SPAMBASE"]
    higgs_formula = None

    print("\n" + "="*60)
    print("🚀 NEURO-SYMBOLIC EFFICIENCY BENCHMARK")
    print("="*60)

    for name in datasets:
        print(f"\n--- Processing {name} ---")
        X, y = get_dataset(name, data_dir="data")
        
        # 1. DISCOVERY PHASE
        formula = discover_formula(X, y, seed=42)
        visualize_activation(formula, name, save_dir="results")
        
        # Save HIGGS formula for transfer
        if name == "HIGGS": 
            higgs_formula = formula
        
        # 2. TRAINING PHASE
        print(f"  [Phase 2] Training Models (3 Seeds each)...")
        
        # A. Heavy Baseline
        res = run_experiment(X, y, name, "Heavy", "ReLU")
        final_results.append(res)
        print(f"    Heavy (ReLU): Eff={res['Efficiency']} | AUC={res['AUC']}")
        
        # B. Light Baselines
        for act in ["ReLU", "GELU", "SiLU"]:
            res = run_experiment(X, y, name, "Light", act)
            final_results.append(res)
            print(f"    Light ({act}): Eff={res['Efficiency']} | AUC={res['AUC']}")
            
        # C. Hybrid (Specialist)
        res = run_experiment(X, y, name, "Light", "Hybrid", formula)
        res["Activation"] = "Hybrid (Specialist)"
        final_results.append(res)
        print(f"    Hybrid (Specialist): Eff={res['Efficiency']} | AUC={res['AUC']}")
        
        # D. Hybrid (Transfer) - Only for Forest/Spam
        if name != "HIGGS" and higgs_formula:
            res = run_experiment(X, y, name, "Light", "Hybrid", higgs_formula)
            res["Activation"] = "Hybrid (Transfer)"
            final_results.append(res)
            print(f"    Hybrid (Transfer): Eff={res['Efficiency']} | AUC={res['AUC']}")

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*60)
    
    df = pd.DataFrame(final_results)
    df = df.sort_values(by=["Dataset", "Efficiency"], ascending=[True, False])
    
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
        
    df.to_csv("results/final_efficiency_results.csv", index=False)
    print("\n💾 Results saved to results/final_efficiency_results.csv")
