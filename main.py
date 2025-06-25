from MSIncClustSolver import MSIncClustSolver
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plt
import pandas as pd
from config import CONFIG

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
A = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
A = StandardScaler().fit_transform(A)
solver = MSIncClustSolver(CONFIG)
centers = solver.solve(A)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(A[:, 0], A[:, 1], s=30, alpha=0.6, label="Customers")
plt.scatter(centers[:, 0], centers[:, 1], color="red", marker="x", s=100, label="Cluster Centers")
labels, _ = pairwise_distances_argmin_min(A, centers)
plt.scatter(A[:, 0], A[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("MSInc-CLUST Clustering on Mall Customers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

