import numpy as np
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicClassifier
from .utils import set_seed

def discover_formula(X, y, seed=42):
    """
    Runs Genetic Programming to find a symbolic activation.
    Returns: formula string
    """
    set_seed(seed)
    
    # Subsample for discovery (10%)
    idx_disc = np.random.choice(len(X), int(0.1 * len(X)), replace=False)
    X_disc, y_disc = X[idx_disc], y[idx_disc]
    s_disc = StandardScaler()
    X_disc = s_disc.fit_transform(X_disc)
    
    print(f"  [Phase 1] Genetic Discovery running...")
    gp = SymbolicClassifier(
        generations=5, 
        population_size=500, 
        function_set=['add', 'sub', 'mul', 'sin', 'cos', 'abs'],
        random_state=seed, 
        n_jobs=-1,
        verbose=0
    )
    gp.fit(X_disc, y_disc)
    formula = str(gp._program)
    print(f"  > Discovered: {formula}")
    return formula
