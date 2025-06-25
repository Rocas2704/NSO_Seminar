CONFIG = {
    "norm": "linf",
    "num_clusters": 5,
    "gamma1": 0.9,
    "gamma2": 0.9,
    "gamma3": 1.1,
    "method":"smoothing",
    "lmbm": {
        "epsilon": 1e-4,
        "max_corrections": 10,
        "epsilon_L": 0.01,
        "epsilon_R": 0.05,
        "kappa": 0.4,
        "gamma":1e-4,
        "omega":2.0,
        "theta":1.0
    },
    "smoothing": {
        "tau_start": 0.2,
        "tau_decay": 0.5,
        "epsilon_start": 1e-2,
        "epsilon_decay": 0.5,
        "max_iter": 15,
        "refine_final_solution": True
    }
}