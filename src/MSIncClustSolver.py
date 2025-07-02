import numpy as np

class MSIncClustSolver:
    def __init__(self, config: dict):
        self.k = config["num_clusters"]
        self.norm = config["norm"]
        self.gamma1 = config["gamma1"]
        self.gamma2 = config["gamma2"]
        self.gamma3 = config["gamma3"]
        self.method = config["method"]
        self.config = config

    def solve(self, A: np.ndarray) -> np.ndarray:
        m, n = A.shape
        centers = [np.mean(A, axis=0)]  # x1 as centroid

        for l in range(2, self.k + 1):
            A5 = self._generate_starting_points(A, centers)

            best_y = None
            best_val = np.inf

            for y0 in A5:
                x_init = np.array(centers + [y0])  # l x n
                x0 = x_init.flatten()
                if self.method == "lmbm":
                    from src.LMBMSolver import LMBMSolver
                    from objective import ClusteringObjective
                    solver = LMBMSolver(self.config)
                    obj = ClusteringObjective(A, len(centers) + 1, self.norm)
                    x_opt, f_val, _ = solver.solve(obj, x0)
                    y_star = x_opt.reshape((len(centers) + 1, -1))[-1]
                elif self.method == "smoothing":
                    from src.HSMSolver import SmoothingSolver
                    from src.HSMSolver import SmoothedClusterObjective  # (k x n)
                    centers_full = np.array(centers + [y0])  # (k x n)
                    obj = SmoothedClusterObjective(
                        A=A,
                        k=len(centers) + 1,
                        centers=centers_full,
                        tau=self.config["smoothing"]["tau_start"],
                        norm=self.norm
                    )

                    solver = SmoothingSolver(self.config)
                    x_opt = solver.solve_objective(obj, x0)
                    f_val = obj(x_opt)
                    y_star = x_opt.reshape((len(centers) + 1, -1))[-1]
                else:
                    raise ValueError("Unsupported method")

                if f_val < best_val:
                    best_val = f_val
                    best_y = y_star

            print(f"Cluster {l}: evaluated {len(A5)} candidates → best value: {best_val:.4f}")
            centers.append(best_y)
            print(f"Nuevos centros hasta k={l}:\n", np.array(centers))

        return np.array(centers)

    def _generate_starting_points(self, A: np.ndarray, centers: list[np.ndarray]) -> list[np.ndarray]:
        import numpy as np

        def deduplicate(points, tol=1e-4):
            unique = []
            for p in points:
                if not any(np.linalg.norm(p - q) < tol for q in unique):
                    unique.append(p)
            return unique

        A = np.asarray(A)
        X = np.array(centers)  # shape (l-1, n)
        m, n = A.shape

        # Schritt 1: A0 = A \ S1
        A0 = [a for a in A if not any(np.allclose(a, c) for c in centers)]

        r_prev = np.min(np.linalg.norm(X[:, None, :] - A[None, :, :], axis=2), axis=0)  # shape (m,)

        # Schritt 2: z_l(a) für alle a in A0
        def z_l(y):
            dists = np.linalg.norm(y - A, axis=1)  # shape (m,)
            return np.mean(np.maximum(0, r_prev - dists))

        scores = [(a, z_l(a)) for a in A0]
        z_max = max(s for _, s in scores)
        A1 = [a for a, s in scores if s >= self.gamma1 * z_max]

        # Schritt 3–4: Repräsentantenbildung
        def influence_set(y):
            return [
                A[i] for i in range(len(A))
                if np.linalg.norm(y - A[i]) < r_prev[i]
            ]

        A2 = []
        for a in A1:
            B3 = influence_set(a)
            if len(B3) == 0:
                continue
            if self.norm == "l2":
                rep = np.mean(B3, axis=0)
            else:
                rep = min(
                    B3,
                    key=lambda x: np.mean([np.linalg.norm(x - b, ord=1 if self.norm == "l1" else np.inf) for b in B3])
                )
            A2.append(rep)

        A2 = deduplicate(A2)

        # Schritt 5: Filter mit gamma2
        scores2 = [(c, z_l(c)) for c in A2]
        z2_max = max(s for _, s in scores2)
        A3 = [c for c, s in scores2 if s >= self.gamma2 * z2_max]

        # Schritt 6–7: Lösung des ACP und Auswahl
        A4 = []
        vals = []
        for c in A3:
            if self.method == "lmbm":
                from src.LMBMSolver import LMBMSolver
                from objective import AuxiliaryObjective
                solver = LMBMSolver(self.config)
                obj = AuxiliaryObjective(A, np.array(centers), self.norm)
                y_star, f_val, _ = solver.solve(obj, c)

            elif self.method == "smoothing":
                from src.HSMSolver import SmoothingSolver
                from src.HSMSolver import SmoothedAuxiliaryObjective
                solver = SmoothingSolver(self.config)
                obj = SmoothedAuxiliaryObjective(
                    A,
                    np.array(centers),
                    tau=self.config["smoothing"]["tau_start"],
                    norm=self.norm
                )
                # Perturba aquí directamente
                c_perturbed = c + np.random.normal(scale=1e-2, size=c.shape)
                y_star = solver.solve_objective(obj, c_perturbed)
                f_val = obj(y_star)

            else:
                raise ValueError("Unsupported method")

            A4.append((y_star, f_val))
            vals.append(f_val)

        fmin = min(vals)
        A5 = [y for (y, val) in A4 if val <= self.gamma3 * fmin]
        A5 = deduplicate(A5)

        print("Final candidates  A5:", len(A5))
        print("r_prev mean:", np.mean(r_prev), "std:", np.std(r_prev))
        print(f"A0: {len(A0)}, A1: {len(A1)}, A2: {len(A2)}, A3: {len(A3)}, A5: {len(A5)}")

        return A5
