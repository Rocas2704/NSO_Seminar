# IS-CLUST: Nonsmooth Clustering with LMBM and HSM

This repository implements the incremental clustering algorithm **IS-CLUST**, combining the *Minimum Sum-of-Distances Clustering* approach with two different nonsmooth optimization methods:

- **LMBM** (Limited Memory Bundle Method)
- **HSM** (Hyperbolic Smoothing Method)

The code is modular and structured for experimentation and performance comparisons between these solvers.

---

## 📁 Project Structure

- `MSIncClustSolver.py`  
  Contains the core implementation of the **MSInc-CLUST** algorithm. For each cluster increment:
  - Generates good candidate points (A₅) via an auxiliary optimization problem.
  - Selects the best candidate via the full clustering objective.
  - Supports both LMBM and HSM as underlying optimization strategies.

- `LMBMSolver.py`  
  Custom implementation of the Limited Memory Bundle Method, capable of solving nonsmooth, nonconvex problems using:
  - Serious/null step control
  - Subgradient aggregation
  - Limited-memory quasi-Newton updates (L-BFGS and SR1)

- `objective.py`  
  - `ClusteringObjective`: Full nonsmooth clustering loss + subgradient  
  - `AuxiliaryObjective`: Simplified (auxiliary) clustering objective used to filter candidate centers

- `HSMSolver.py`  
  - `SmoothedClusterObjective`: Smooth version of the full clustering problem (used in HSM)  
  - `SmoothedAuxiliaryObjective`: Smooth version of the auxiliary problem  
  - `SmoothingSolver`: τ-scheduling wrapper using L-BFGS-B

---

## 🧪 How It Works

### Step-by-step Process

1. **Initialization**: The first center is the centroid of all data points.
2. **Incremental Loop (for `k` clusters)**:
   - Generate candidate points from `A₀` using local distances.
   - Filter and refine candidates through an auxiliary optimization (solved with LMBM or HSM).
   - Evaluate the full clustering objective for all `A₅` candidates.
   - Choose the best-performing center and append it.
3. **(Optional)**: Run a final refinement with the full clustering objective over all centers.

---

## 🛠 Configuration Example

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
