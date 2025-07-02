import time
import numpy as np
from scipy.optimize import minimize
import time

class LMBMSolver:
    def __init__(self, config:dict):
        self.epsilon = config["lmbm"]["epsilon"]
        self.max_corrections = config["lmbm"]["max_corrections"]
        self.epsilon_L = config["lmbm"]["epsilon_L"]
        self.epsilon_R = config["lmbm"]["epsilon_R"]
        self.kappa = config["lmbm"]["kappa"]
        self.gamma = config["lmbm"]["gamma"]
        self.omega = config["lmbm"]["omega"]
        self.theta = config["lmbm"]["theta"]
        self.last_step = "serious" # to distinguish step types

    def line_search(self, objective_fn, x, d, f_x, xi_tilde, beta_tilde):
        # Custom backtracking line search with interpolation
        t_L = 0.0
        t_U = 1.0
        t = 1.0
        w_k = -xi_tilde @ d + 2 * beta_tilde
        max_iter = 30
        i=0

        while True:
            y = x + t * d
            f_y, xi_y = objective_fn(y)
            diff = y - x
            beta = max(
                abs(f_x - f_y + (y - x) @ xi_y),
                self.gamma * (np.linalg.norm(diff) ** self.omega
            ))

            # Serious Step
            if f_y <= f_x - self.epsilon_L * t * w_k:
                return y, f_y, xi_y, 0.0, t  # serious step

            # Null Step
            elif -beta + d @ xi_y >= self.epsilon_R * w_k:
                return y, f_y, xi_y, beta, t  # null step

            # Interpolation step
            t_L = t
            t_U = t if t == t_U else t_U
            t = np.random.uniform(
                t_L + self.kappa * (t_U - t_L),
                t_U - self.kappa * (t_U - t_L)

            )
            i += 1
            if i >= max_iter:
                print(f"[LMBM] Line search max iterations reached at t={t:.2e}")
                return y, f_y, xi_y, beta, t


    def _aggregate_subgradients(self, xi_m, xi_new, xi_prev, beta_new, beta_prev, D):
        """
            Performs subgradient aggregation according to the theoretical formulation of LMBM.
            Minimizes the function:
                phi(λ) = ||λ₁·xi_m + λ₂·xi_new + λ₃·xi_prev||²_D + 2(λ₂·beta_new + λ₃·beta_prev)
            subject to: λ₁ + λ₂ + λ₃ = 1, λᵢ >= 0

            Returns:
                xi_tilde_next: new aggregated subgradient
                beta_tilde_next: new aggregated beta_tilde
        """
        xi_m = xi_m.reshape(-1,1)
        xi_new = xi_new.reshape(-1,1)
        xi_prev = xi_prev.reshape(-1,1)

        def objective(lam):
            lam1, lam2, lam3 = lam
            s = lam1 * xi_m + lam2 * xi_new + lam3 * xi_prev
            norm_term = float(s.T @ D @ s)
            beta_term = 2 * (lam2 * beta_new + lam3 * beta_prev)
            return norm_term + beta_term

        cons = ({'type': 'eq', 'fun': lambda l: np.sum(l) - 1},
                {'type': 'ineq', 'fun': lambda l: l[0]},
                {'type': 'ineq', 'fun': lambda l: l[1]},
                {'type': 'ineq', 'fun': lambda l: l[2]})

        res = minimize(objective, x0=[1 / 3, 1 / 3, 1 / 3], constraints=cons)

        if not res.success:
            raise RuntimeError("Aggregated subgradient optimization failed")

        lam_opt = res.x
        xi_tilde_next = lam_opt[0] * xi_m + lam_opt[1] * xi_new + lam_opt[2] * xi_prev
        beta_tilde_next = lam_opt[1] * beta_new + lam_opt[2] * beta_prev

        return xi_tilde_next.flatten(), beta_tilde_next

    def _apply_inverse_lbfgs(self, v, S_list, U_list, theta):
        if len(S_list) == 0:
            return theta * v

        S = np.column_stack([np.asarray(s).reshape(-1) for s in S_list])
        U = np.column_stack([np.asarray(u).reshape(-1) for u in U_list])
        m = S.shape[1]

        R = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                R[i, j] = S[:, i].T @ U[:, j]

        C = np.diag([S[:, i].T @ U[:, i] for i in range(m)])

        R_inv = np.linalg.pinv(R)
        M1 = np.block([
            [R_inv.T @ C + theta * U.T @ U @ R_inv, -(R_inv.T)],
            [-R_inv, np.zeros_like(R)]
        ])
        W = np.hstack([S, theta * U])

        return theta * v + W @ (M1 @ (W.T @ v))

    def _apply_inverse_sr1(self, v, S_list, U_list, theta):
        # If not enough correction pairs, return scaled identity
        if len(S_list) < 2:
            return theta * v

        # Stack correction pairs into matrices of shape (n, m)
        S = np.column_stack([np.asarray(s).reshape(-1) for s in S_list])
        U = np.column_stack([np.asarray(u).reshape(-1) for u in U_list])

        # Compute R and C matrices
        R = S.T @ U  # shape (m, m)
        C = np.diag([S[:, i] @ U[:, i] for i in range(S.shape[1])])  # shape (m, m)

        # Compute inverse SR1 components
        term1 = theta * np.eye(v.shape[0])
        term2 = theta * (U.T @ U) - R - R.T + C

        # Ensure middle is at least 2D to prevent shape errors
        middle = np.linalg.pinv(term2)
        diff = theta * U - S  # (n, m)
        correction = diff @ middle @ diff.T  # (n, n)

        return (term1 - correction) @ v

    def solve(self, objective_fn, x0:np.ndarray):
            x = x0.copy()
            f_val, xi = objective_fn(x)

            xi_tilde = xi.copy()
            xi_m = xi.copy()
            beta_tilde = 0.0
            h = 0
            prev_f_val = f_val
            stagnation = 0
            S, U = [],[]
            D = np.eye(x.shape[0])
            d = -D@xi_tilde
            MAX_ITER = 100

            start_time = time.time()
            TIME_LIMIT =10

            while True:
                w_k = -xi_tilde@d +2*beta_tilde
                q_k = 0.5*np.linalg.norm(xi_tilde)**2 + beta_tilde

                if h % 10 == 0:
                    print(f"[LMBM] Iter {h}: w_k = {w_k:.2e}, q_k = {q_k:.2e}, f_val = {f_val:.4f}")

                if w_k < self.epsilon and q_k < 100 * self.epsilon:
                    print(f"[LMBM] Converged at iteration {h}")
                    break

                if h >= MAX_ITER:
                    print("[LMBM] Max iterations reached, aborting.")
                    break

                if np.linalg.norm(xi_tilde) < 1e-6:
                    print("[LMBM] Subgradient too flat, aborting.")
                    break
                if time.time() - start_time > TIME_LIMIT:
                    print("[LMBM] Time limit reached, stopping.")
                    break


                y_next, f_next, xi_next, beta_next, t_used = self.line_search(objective_fn, x, d, f_val, xi_tilde,
                                                                              beta_tilde)
                print(f"[LMBM] y_next[:3] = {y_next[:3]}, f_next = {f_next:.4f}")
                if f_next <= f_val - self.epsilon_L * t_used * w_k:
                    # Serious step
                    self.last_step = "serious"
                    s = y_next - x
                    u = xi_next - xi_m
                    S.append(s)
                    U.append(u)
                    if len(S) > self.max_corrections:
                        S.pop(0)
                        U.pop(0)
                    x = y_next
                    f_val = f_next
                    xi_m = xi_next.copy()
                    beta_tilde = 0.0
                    xi_tilde = xi_next.copy()
                else:
                    # Null step
                    self.last_step = "null"
                    xi_tilde, beta_tilde = self._aggregate_subgradients(
                        xi_m, xi_next, xi_tilde, beta_next, beta_tilde, D
                    )
                if abs(f_val - prev_f_val) < 1e-6:
                    stagnation += 1
                if stagnation > 5:
                    print("[LMBM] Stopped because of repetitive stagnation")
                    break
                else:
                    stagnation = 0

                prev_f_val = f_val

                if len(S) == 0:
                    d = -self.theta * xi_tilde
                else:
                    if self.last_step == "serious":
                        d = -self._apply_inverse_lbfgs(xi_tilde, S, U, self.theta)
                    else:
                        d = -self._apply_inverse_sr1(xi_tilde, S, U, self.theta)

                h += 1

            return x, f_val,h

