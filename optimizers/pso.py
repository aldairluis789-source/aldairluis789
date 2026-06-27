import numpy as np
from models.equilibrium import objective_function

def pso_lazzus_optimizer(bounds, x_exp, y_exp, P_total, model_type="NRTL", config=None):
    if config is None:
        config = {
            'N_PART': 250, 'K_MAX': 1000, 'C1': 1.494, 'C2': 1.494,
            'V_MAX': 12.0, 'W_MAX': 0.7, 'W_MIN': 0.5
        }

    dim = len(bounds)
    N_PART = config['N_PART']
    K_MAX = config['K_MAX']
    V_MAX = config['V_MAX']

    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(N_PART, dim))
    V = np.random.uniform(-V_MAX, V_MAX, size=(N_PART, dim))

    p_best = np.copy(X)
    f_p_best = np.array([objective_function(X[i], x_exp, y_exp, P_total, model_type) for i in range(N_PART)])

    g_best_idx = np.argmin(f_p_best)
    g_best = np.copy(p_best[g_best_idx])
    f_g_best = f_p_best[g_best_idx]

    print("\nIniciando optimización por enjambre de partículas...")

    for k in range(1, K_MAX + 1):
        w = config['W_MAX'] - ((config['W_MAX'] - config['W_MIN']) / K_MAX) * k

        for i in range(N_PART):
            r1 = np.random.uniform(0, 1, dim)
            r2 = np.random.uniform(0, 1, dim)

            V[i] = w * V[i] + config['C1'] * r1 * (p_best[i] - X[i]) + config['C2'] * r2 * (g_best - X[i])
            V[i] = np.clip(V[i], -V_MAX, V_MAX)

            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], bounds[:, 0], bounds[:, 1])

            f_current = objective_function(X[i], x_exp, y_exp, P_total, model_type)

            if f_current < f_p_best[i]:
                f_p_best[i] = f_current
                p_best[i] = np.copy(X[i])

                if f_current < f_g_best:
                    f_g_best = f_current
                    g_best = np.copy(X[i])

        if k % 50 == 0 or k == 1:
            print(f" > Generación {k:4d}/{K_MAX} -> Valor de OF mínimo: {f_g_best:.6f}")

    return g_best, f_g_best