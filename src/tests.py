import pandas as pd
import numpy as np
from time import time
import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("output", exist_ok=True)

from src.MSIncClustSolver import MSIncClustSolver
from src.config import CONFIG



# === Loaders ===
def load_mall_data():
    df = pd.read_csv("data/Mall_Customers.csv")
    df = df.select_dtypes(include=[np.number])  # Only numeric columns
    return StandardScaler().fit_transform(df)

def load_segmentation_data():
    df = pd.read_csv("data/segmentation.data", skiprows=5, header=None)
    df = df.drop(columns=[0])  # drop label column
    return StandardScaler().fit_transform(df.values)

def load_online_news_data():
    df = pd.read_csv("data/OnlineNewsPopularity.csv")
    df = df.drop(columns=[col for col in ['url', ' shares', 'shares'] if col in df.columns])
    return StandardScaler().fit_transform(df.values)

# === General clustering wrapper for all methods ===
def clustering_wrapper(X, k, method, gamma_params=None):
    if method in ["lmbm", "smoothing"]:
        import copy
        local_config = copy.deepcopy(CONFIG)
        local_config["num_clusters"] = k
        local_config["method"] = method
        if gamma_params:
            local_config["gamma1"], local_config["gamma2"], local_config["gamma3"] = gamma_params
        solver = MSIncClustSolver(local_config)
        centers = solver.solve(X)
        labels = np.argmin(
            np.linalg.norm(X[:, None, :] - centers[None, :, :],
                           ord=1 if CONFIG["norm"] == "l1"
                           else np.inf if CONFIG["norm"] == "linf" else 2, axis=2),
            axis=1
        )
        return centers, labels

    elif method == "kmeans":
        model = KMeans(n_clusters=k, init="random", n_init= 1, random_state=42)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        return centers, labels

    else:
        raise ValueError("Unsupported method.")


# === Compute objective value ===
def compute_objective(X, centers):
    p = CONFIG["norm"]
    if p == "l1":
        norm = 1
    elif p == "l2":
        norm = 2
    elif p == "linf":
        norm = np.inf
    else:
        raise ValueError("Unsupported norm.")

    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], ord=norm, axis=2)
    min_dist = np.min(distances, axis=1)
    return np.mean(min_dist)


# === Run test and compute metrics ===
def run_test(algorithm, X, k, method, gamma_params=None):
    start_time = time()
    centers, labels = algorithm(X, k, method, gamma_params)
    runtime = time() - start_time
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    obj_value = compute_objective(X, centers)
    return {
        'method': method,
        'k': k,
        'time': runtime,
        'silhouette': sil,
        'db_index': db,
        'objective': obj_value
    }


# === Main execution ===
# === Main execution ===
def run_clustering_for_dataset(dataset_name, X, output_filename):
    norms = ["l1", "l2", "linf"]
    methods = ["lmbm", "kmeans", "smoothing"]
    ks = [3, 5, 7]
    gamma_params = (0.975, 0.99, 1.001)
    results = []
    stored_labels = {}

    for norm in norms:
        CONFIG["norm"] = norm
        print(f"\n=== Norm: {norm} ===")
        for method in methods:
            print(f"=== Method: {method} ===")
            for k in tqdm(ks, desc=f"{method}-{dataset_name}-{norm}"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Running for k = {k}...")
                centers, labels = clustering_wrapper(X, k, method, gamma_params if method != "kmeans" else None)
                sil = silhouette_score(X, labels)
                db = davies_bouldin_score(X, labels)
                obj = compute_objective(X, centers)
                runtime = time()
                results.append({
                    'dataset': dataset_name,
                    'method': method,
                    'k': k,
                    'time': runtime,
                    'silhouette': sil,
                    'db_index': db,
                    'norm': norm
                })
                if k == 5:
                    stored_labels[(method, norm)] = labels

    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)

    # === Plots ===
    sns.set(style="whitegrid")
    for metric in ["silhouette", "db_index", "time"]:
        for norm in df["norm"].unique():
            subset = df[df["norm"] == norm]
            plt.figure(figsize=(12, 4))
            sns.lineplot(data=subset, x="k", y=metric, hue="method", marker="o")
            plt.title(f"{metric.replace('_', ' ').title()} vs. k – {dataset_name} (Norm: {norm})")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.legend(title="Method")
            plt.tight_layout()
            plt.savefig(f"output/{dataset_name}_{metric}_vs_k_{norm}.png")
            plt.show()

    # === PCA visualization for k=5 ===
    X_pca = PCA(n_components=2).fit_transform(X)
    for norm in norms:
        plt.figure(figsize=(16, 5))
        for i, method in enumerate(methods):
            labels = stored_labels.get((method, norm), None)
            if labels is None:
                continue
            plt.subplot(1, len(methods), i + 1)
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=25, legend=False)
            plt.title(f"{method.capitalize()} – Norm: {norm}")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
        plt.suptitle(f"{dataset_name}: PCA-Projektion bei Norm {norm.upper()} (k=5)")
        plt.tight_layout()
        plt.savefig(f"output/{dataset_name}_pca_{norm}_k5.png")

        plt.show()

# === Main ===
if __name__ == "__main__":
    datasets = {
        "mall": (load_mall_data, "all_clustering_results_small_datasets.csv"),
        "segmentation": (load_segmentation_data, "all_clustering_results_medium_datasets.csv"),
        "online_news": (load_online_news_data, "all_clustering_results_large_datasets.csv")
    }

    for name, (loader, output_filename) in datasets.items():
        print(f"\n>>> Running for dataset: {name.upper()}")
        X = loader()
        run_clustering_for_dataset(name, X, f"output/{output_filename}")