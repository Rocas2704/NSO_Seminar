import numpy as np

class ClusteringObjective:
    def __init__(self, A:np.ndarray, k: int, norm: str):
        self.A = A
        self.k = k
        self.norm = norm
        self.m, self.n = A.shape
    def __call__(self, x:np.ndarray) -> tuple[float, np.ndarray]:
        centers = x.reshape((self.k, self.n))
        grad = np.zeros_like(centers)
        f_val = 0.0

        for a in self.A:
            if self.norm =="l1":
                dists = np.sum(np.abs(centers - a), axis=1)
            elif self.norm == "l2":
                dists = np.sum((centers - a)**2, axis=1)
            elif self.norm == "linf":
                dists = np.max(np.abs(centers - a), axis =1)
            else:
                raise ValueError(f"Invalid norm{self.norm}")
            min_dist = np.min(dists)
            f_val+= min_dist
            active = np.where(np.abs(dists - min_dist) < 1e-8)[0]
            weights = np.ones(len(active)) / len(active)

            for idx, j in enumerate(active):
                diff = centers[j] - a
                if self.norm == "l2":
                    # Gradient of ||x - a|| is (x - a)/||x - a||, or 2(x - a) for ||x - a||^2
                    norm = np.linalg.norm(diff)
                    if norm > 1e-8:
                        grad[j] += diff / norm * weights[idx]
                elif self.norm == "l1":
                    # Subgradient as in (4.4)
                    g = np.sign(diff)
                    zero_mask = (diff == 0)
                    g[zero_mask] = np.random.uniform(-1, 1, size=g[zero_mask].shape)

                    grad[j] += g * weights[idx]
                elif self.norm == "linf":
                    # Subgradient as in (4.9)
                    abs_diff = np.abs(diff)
                    max_val = np.max(abs_diff)
                    active_coords = np.where(abs_diff == max_val)[0]
                    e = np.zeros(self.n)
                    q = active_coords[0]  # pick one coordinate
                    e[q] = np.sign(diff[q])
                    grad[j] += e * weights[idx]

        return f_val / self.m, grad.flatten() / self.m

class AuxiliaryObjective:
    def __init__(self, A: np.ndarray, current_centers: np.ndarray, norm: str = "l2"):
        self.A = A
        self.X = current_centers  # (k-1) x n
        self.norm = norm
        self.m, self.n = A.shape

        # Precompute r_k-1^i: min distance to existing centers
        if self.norm == "l2":
            self.r_prev = np.array([
                np.min(np.linalg.norm(self.X - a_i, axis=1))
                for a_i in self.A
            ])
        elif self.norm == "l1":
            self.r_prev = np.array([
                np.min(np.sum(np.abs(self.X - a_i), axis=1))
                for a_i in self.A
            ])
        elif self.norm == "linf":
            self.r_prev = np.array([
                np.min(np.max(np.abs(self.X - a_i), axis=1))
                for a_i in self.A
            ])
        else:
            raise ValueError(f"Invalid norm: {self.norm}")

    def __call__(self, y: np.ndarray) -> tuple[float, np.ndarray]:
        # y: shape (n,)
        diffs = self.A - y  # shape (m, n)
        if self.norm == "l2":
            dists = np.linalg.norm(diffs, axis=1)  # shape (m,)
        elif self.norm == "l1":
            dists = np.sum(np.abs(diffs), axis=1)
        elif self.norm == "linf":
            dists = np.max(np.abs(diffs), axis=1)
        else:
            raise ValueError(f"Invalid norm: {self.norm}")

        mask = dists < self.r_prev
        f_val = np.sum(np.where(mask, dists, self.r_prev))

        grad = np.zeros_like(y)
        if self.norm == "l2":
            # Solo usar puntos donde d < r_prev y d > 0 para evitar división por cero
            valid = mask & (dists > 1e-8)
            grad = np.sum((diffs[valid].T / dists[valid]).T, axis=0)
        elif self.norm == "l1":
            valid_diffs = diffs[mask]
            g = np.sign(valid_diffs)
            zero_mask = (valid_diffs == 0)
            g[zero_mask] = np.random.uniform(-1, 1, size=g[zero_mask].shape)
            grad = np.sum(g, axis=0)
        elif self.norm == "linf":
            for i in np.where(mask)[0]:
                diff = diffs[i]
                abs_diff = np.abs(diff)
                max_val = np.max(abs_diff)
                active_coords = np.where(abs_diff == max_val)[0]
                e = np.zeros(self.n)
                e[active_coords[0]] = np.sign(diff[active_coords[0]])
                grad += e

        return f_val / self.m, grad / self.m






