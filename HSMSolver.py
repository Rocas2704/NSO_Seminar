import numpy as np
from scipy.optimize import minimize

class SmoothedClusterObjective:
    def __init__(self, A:np.ndarray, centers :np.ndarray, k:int, tau:float, norm :str):
        self.A = A
        self.k = k
        self.tau = tau
        self.norm = norm
        self.m, self.n = A.shape
        self.t = self._compute_t(centers)

    def update_tau(self, new_tau: float):
        self.tau = new_tau

    def _compute_t(self, centers: np.ndarray):
        t = np.zeros(self.m)
        for i in range(self.m):
            a_i = self.A[i]
            if self.norm == "l2":
                dists = np.linalg.norm(centers - a_i, axis=1)
            elif self.norm == "l1":
                dists = np.sum(np.abs(centers - a_i), axis=1)
            elif self.norm == "linf":
                dists = np.max(np.abs(centers - a_i), axis=1)
            else:
                raise ValueError("Unsupported norm")
            t[i] = -np.min(dists)
        return t
    def _dist(self, xj:np.ndarray, ai: np.ndarray) -> float:
        if self.norm == "l2":
            return np.linalg.norm(xj - ai)
        elif self.norm == "l1":
            return np.sum(np.sqrt((xj - ai) ** 2 + self.tau ** 2))
        elif self.norm == "linf":
            theta = np.max(np.abs(xj - ai))
            omega = np.concatenate([xj - ai, -ai + xj])
            term = omega - theta
            return theta + 0.5 * np.sum(term + np.sqrt(term ** 2 + self.tau ** 2))
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

    def __call__(self, x: np.ndarray) -> float:
        centers = x.reshape((self.k, self.n))
        total = 0.0

        for i in range(self.m):
            ti = self.t[i]
            sum_term = 0.0
            for j in range(self.k):
                d = self._dist(centers[j], self.A[i])
                z = ti + d
                sum_term += (z - np.sqrt(z ** 2 + self.tau ** 2)) / 2
            total += -ti + sum_term

        return total / self.m


class SmoothedAuxiliaryObjective:
    def __init__(self, A: np.ndarray, X_prev: np.ndarray, tau: float, norm: str):
        self.A = A
        self.X_prev = X_prev  # (k-1) x n array of previous centers
        self.tau = tau
        self.norm = norm
        self.m, self.n = A.shape

        # Precompute r_{k-1}^i: minimum distance to previous centers
        self.r_prev = []
        for ai in self.A:
            if norm == "l2":
                dists = np.linalg.norm(self.X_prev - ai, axis=1)
            elif norm == "l1":
                dists = np.sum(np.sqrt((self.X_prev - ai)**2 + tau**2), axis=1)
            elif norm == "linf":
                dists = np.max(np.abs(self.X_prev - ai), axis=1) + tau
            else:
                raise ValueError("Unsupported norm")
            self.r_prev.append(np.min(dists))
        self.r_prev = np.array(self.r_prev)

    def _dist(self, y: np.ndarray, ai: np.ndarray) -> float:
        if self.norm == "l2":
            return np.linalg.norm(y - ai)
        elif self.norm == "l1":
            return np.sum(np.sqrt((y - ai)**2 + self.tau**2))
        elif self.norm == "linf":
            theta = np.max(np.abs(y - ai))
            omega = np.concatenate([y - ai, -ai + y])
            term = omega - theta
            return theta + 0.5 * np.sum(term + np.sqrt(term**2 + self.tau**2))
        else:
            raise ValueError("Unsupported norm")

    def __call__(self, y: np.ndarray) -> float:
        total = 0.0
        for i in range(self.m):
            d = self._dist(y, self.A[i])
            r_i = self.r_prev[i]
            term = r_i + d - 0.5 * np.sqrt((r_i - d)**2 + self.tau**2)
            total += term
        return total / self.m
    def update_tau(self, new_tau):
        self.tau = new_tau

class SmoothingSolver:
    def __init__(self, config: dict):
        self.tau = config["smoothing"]["tau_start"]
        self.tau_decay = config["smoothing"]["tau_decay"]
        self.epsilon = config["smoothing"]["epsilon_start"]
        self.epsilon_decay = config["smoothing"]["epsilon_decay"]
        self.max_iter = config["smoothing"]["max_iter"]

    def solve_objective(self, objective_fn, x0: np.ndarray) -> np.ndarray:
        """
        Minimizes a smooth objective function (with no explicit gradient).
        Returns the final solution.
        """
        x = x0.copy()
        tau = self.tau
        epsilon = self.epsilon

        for _ in range(self.max_iter):
            result = minimize(lambda x_: objective_fn(x_), x, method="L-BFGS-B")
            x = result.x

            # Approximate gradient norm (finite difference or just re-eval)
            f0 = objective_fn(x)
            grad_norm = np.linalg.norm(x - x0) / max(1e-8, np.abs(f0))
            if grad_norm < epsilon:
                break

            tau *= self.tau_decay
            epsilon *= self.epsilon_decay
            objective_fn.update_tau(tau)
            x_prev = x.copy()

        return x




