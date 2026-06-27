import numpy as np

def nrtl_gamma(x, T, B_ij, B_ji, alpha):
    R = 1.9872
    tau_ij = B_ij / (R * T)
    tau_ji = B_ji / (R * T)

    G_ij = np.exp(-alpha * tau_ij)
    G_ji = np.exp(-alpha * tau_ji)

    x1, x2 = x[0], x[1]

    ln_gamma1 = (x2**2) * (tau_ji * (G_ji / (x1 + x2 * G_ji))**2 + (tau_ij * G_ij) / (x1 * G_ij + x2)**2)
    ln_gamma2 = (x1**2) * (tau_ij * (G_ij / (x2 + x1 * G_ij))**2 + (tau_ji * G_ji) / (x2 * G_ji + x1)**2)

    return np.exp(ln_gamma1), np.exp(ln_gamma2)

def uniquac_gamma(x, T, U_ij, U_ji, r1=2.1055, q1=1.972, r2=0.9200, q2=1.400):
    R = 1.9872
    x1, x2 = x[0], x[1]

    tau_ij = np.exp(-U_ij / (R * T))
    tau_ji = np.exp(-U_ji / (R * T))

    sum_xr = x1 * r1 + x2 * r2
    Phi1 = (x1 * r1) / sum_xr
    Phi2 = (x2 * r2) / sum_xr

    sum_xq = x1 * q1 + x2 * q2
    theta1 = (x1 * q1) / sum_xq
    theta2 = (x2 * q2) / sum_xq

    z = 10.0
    l1 = (z / 2.0) * (r1 - q1) - (r1 - 1.0)
    l2 = (z / 2.0) * (r2 - q2) - (r2 - 1.0)

    ln_gamma1_C = np.log(Phi1 / x1) + (z / 2.0) * q1 * np.log(theta1 / Phi1) + Phi2 * (l1 - (r1 / r2) * l2)
    ln_gamma2_C = np.log(Phi2 / x2) + (z / 2.0) * q2 * np.log(theta2 / Phi2) + Phi1 * (l2 - (r2 / r1) * l1)

    ln_gamma1_R = -q1 * np.log(theta1 + theta2 * tau_ji) + q1 * theta2 * (tau_ji / (theta1 + theta2 * tau_ji) - tau_ij / (theta2 + theta1 * tau_ij))
    ln_gamma2_R = -q2 * np.log(theta2 + theta1 * tau_ij) + q2 * theta1 * (tau_ij / (theta2 + theta1 * tau_ij) - tau_ji / (theta1 + theta2 * tau_ji))

    return np.exp(ln_gamma1_C + ln_gamma1_R), np.exp(ln_gamma2_C + ln_gamma2_R)