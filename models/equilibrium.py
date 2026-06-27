import numpy as np
from models.activity_coefficients import nrtl_gamma, uniquac_gamma

def bubble_point_solver(x1, P_total, params, model_type="NRTL"):
    x = [x1, 1.0 - x1]

    A1, B1, C1_ant = 16.8958, 3795.17, -42.23
    A2, B2, C2_ant = 16.3872, 3885.70, -42.98

    def evaluar_presion(T_k):
        if T_k + C1_ant <= 0 or T_k + C2_ant <= 0:
            return 1e10, 1.0, 0.0

        if model_type == "NRTL":
            B_ij, B_ji, alpha = params
            g1, g2 = nrtl_gamma(x, T_k, B_ij, B_ji, alpha)
        elif model_type == "UNIQUAC":
            U_ij, U_ji = params
            g1, g2 = uniquac_gamma(x, T_k, U_ij, U_ji)
        else:
            g1, g2 = 1.0, 1.0

        if np.isnan(g1) or np.isinf(g1) or g1 <= 0: g1 = 1e-5
        if np.isnan(g2) or np.isinf(g2) or g2 <= 0: g2 = 1e-5

        P_sat1 = np.exp(A1 - B1 / (T_k + C1_ant))
        P_sat2 = np.exp(A2 - B2 / (T_k + C2_ant))
        P_calc = x[0] * g1 * P_sat1 + x[1] * g2 * P_sat2

        if np.isnan(P_calc) or np.isinf(P_calc):
            P_calc = 1e10

        return P_calc, g1, P_sat1

    T_min, T_max = 280.0, 500.0
    T_calc = 350.0
    gamma1_final, P_sat1_final = 1.0, 0.0

    for _ in range(35):
        T_mid = (T_min + T_max) / 2.0
        P_calc, g1, ps1 = evaluar_presion(T_mid)

        if P_calc < P_total:
            T_min = T_mid
        else:
            T_max = T_mid

        T_calc = T_mid
        gamma1_final, P_sat1_final = g1, ps1

    y1_calc = (x[0] * gamma1_final * P_sat1_final) / P_total

    if np.isnan(y1_calc) or np.isinf(y1_calc):
        y1_calc = 0.0

    y1_calc = np.clip(y1_calc, 0.0, 1.0)
    T_calc_celsius = T_calc - 273.15
    return T_calc_celsius, y1_calc

def objective_function(params, x_exp, y_exp, P_total, model_type="NRTL"):
    OF = 0.0
    for i in range(len(x_exp)):
        _, y1_calc = bubble_point_solver(x_exp[i], P_total, params, model_type)
        of_puntos = (y_exp[i] - y1_calc) ** 2
        if not np.isnan(of_puntos):
            OF += of_puntos
    return OF