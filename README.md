# Clustering Optimization with IS-CLUST, LMBM and HSM

This repository implements the incremental clustering algorithm **IS-CLUST**, combining the *Minimum Sum-of-Distances Clustering* approach with two different nonsmooth optimization methods:

- **LMBM** (Limited Memory Bundle Method)
- **HSM** (Hyperbolic Smoothing Method)

The code is modular and structured for experimentation and performance comparisons between these solvers.

---

## 📁 Project Structure

- `src/MSIncClustSolver.py`  
  Core implementation of the **MSInc-CLUST** algorithm. For each cluster increment:
  - Generates candidate centers (`A₅`) via an auxiliary optimization problem.
  - Selects the best candidate using the full clustering objective.
  - Supports both LMBM and HSM as solvers.

- `src/LMBMSolver.py`  
  Custom implementation of the Limited Memory Bundle Method:
  - Serious/null step control
  - Subgradient aggregation
  - Limited-memory quasi-Newton updates (L-BFGS or SR1)

- `src/HSMSolver.py`  
  - `SmoothedClusterObjective`: Smooth approximation of the clustering loss  
  - `SmoothingSolver`: τ-decay wrapper using L-BFGS-B

- `src/objective.py`  
  - `ClusteringObjective`: Nonsmooth full loss + subgradient  
  - `AuxiliaryObjective`: Simplified auxiliary clustering loss

- `src/tests.py`  
  Main test script. Loads datasets, runs clustering with various methods (k-means, LMBM, HSM), and generates performance metrics and visualizations.

- `data/`  
  Input datasets: `Mall_Customers.csv`, `OnlineNewsPopularity.csv`, `segmentation.data`, etc.

- `output/`  
  Automatically generated clustering results (CSV) and plots (PNG)

---

## 🧪 How It Works

### Step-by-step Process

1. **Initialization**: Start with the global centroid as the first cluster center.
2. **Incremental Loop for `k` clusters**:
   - Generate local candidates using data point neighborhoods.
   - Refine with an auxiliary clustering objective (solved by LMBM or HSM).
   - Evaluate full loss on candidate set (`A₅`) and pick the best.
3. **Optional refinement**: Perform post-processing over all found centers.

---

## ⚙️ Configuration

Controlled via `CONFIG` in `src/config.py`.

### Example:

```python
CONFIG = {
    "norm": "l2",
    "num_clusters": 5,
    "gamma1": 0.9,
    "gamma2": 0.9,
    "gamma3": 1.1,
    "method": "lmbm",  # or "smoothing"
    "lmbm": {
        "epsilon": 1e-4,
        "max_corrections": 10,
        "epsilon_L": 0.01,
        "epsilon_R": 0.05,
        "kappa": 0.4,
        "gamma": 1e-4,
        "omega": 2.0,
        "theta": 1.0
    },
    "smoothing": {
        "tau_start": 0.2,
        "tau_decay": 0.5,
        "epsilon_start": 1e-2,
        "epsilon_decay": 0.5,
        "max_iter": 15
    }
}

```
## 📦 Requirements

You can install all dependencies via:
pip install -r requirements.txt




## License & Credits

Developed as part of a seminar on continuous optimization for clustering.
Supervised at the Institute for Operations Research (IOR), KIT. by Msc. Stefan Schwarze
