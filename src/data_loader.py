import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import gzip
import shutil
import os

def get_dataset(name, data_dir="data", limit_rows=100000):
    """
    Downloads and preprocesses the dataset.
    Returns: X (features), y (labels)
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n[Data] Fetching {name}...")
    
    if name == "HIGGS":
        csv_path = os.path.join(data_dir, "HIGGS.csv")
        if not os.path.exists(csv_path):
            print("  Downloading HIGGS (This might take a while)...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            gz_path = csv_path + ".gz"
            with requests.get(url, stream=True) as r:
                with open(gz_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        df = pd.read_csv(csv_path, header=None, nrows=limit_rows)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        
    elif name == "FOREST_COVER":
        data = fetch_covtype()
        X = data.data
        # Convert to binary: Spruce/Fir (2) vs Others
        y = (data.target == 2).astype(int) 
        if len(y) > limit_rows:
            idx = np.random.choice(len(y), limit_rows, replace=False)
            X, y = X[idx], y[idx]
            
    elif name == "SPAMBASE":
        data = fetch_openml(name='spambase', version=1, as_frame=False)
        X = data.data
        y = data.target.astype(int)
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    return X, y
