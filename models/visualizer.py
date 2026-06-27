import numpy as np
import matplotlib.pyplot as plt
from models.activity_coefficients import nrtl_gamma, uniquac_gamma
from models.equilibrium import bubble_point_solver, objective_function

def graficar_todo(x_exp, y_exp, T_exp, P_total, mejores_params, model_type="NRTL"):
    # Activar un estilo limpio y profesional para la exposición
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Crear una ventana grande para que se aprecie bien en el proyector
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"ANÁLISIS AVANZADO DE EQUILIBRIO ELV ({model_type}) - OPTIMIZACIÓN PSO (LAZZÚS 2010)", 
                 fontsize=14, fontweight='bold', color='#1a1a1a')

    # =========================================================================
    # 1. DIAGRAMA DE FASE TERMODINÁMICO T-x-y (2D)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    x_calc_line = np.linspace(0.001, 0.999, 100)
    T_calc_line = []
    y_calc_line = []
    
    for xi in x_calc_line:
        Tc, yc = bubble_point_solver(xi, P_total, mejores_params, model_type)
        T_calc_line.append(Tc)
        y_calc_line.append(yc)

    # Puntos experimentales del paper
    ax1.scatter(x_exp, T_exp, color='#e74c3c', s=50, label='T-x Exp (Burbuja)', zorder=5, edgecolor='black')
    ax1.scatter(y_exp, T_exp, color='#3498db', s=50, label='T-y Exp (Rocío)', zorder=5, edgecolor='black')
    
    # Curvas continuas calculadas por tu PSO
    ax1.plot(x_calc_line, T_calc_line, color='#c0392b', lw=2.5, label='Curva de Burbuja (PSO)')
    ax1.plot(y_calc_line, T_calc_line, color='#2980b9', lw=2.5, label='Curva de Rocío (PSO)')
    
    ax1.set_title("Diagrama de Fase T-x1-y1 (Etanol + Agua)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Fracción molar del Etanol ($x_1, y_1$)", fontsize=10)
    ax1.set_ylabel("Temperatura (°C)", fontsize=10)
    ax1.legend(frameon=True, facecolor='white')

    # =========================================================================
    # 2. COEFICIENTES DE ACTIVIDAD (Gama 1 y Gama 2 vs x1)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    gamma1_plot, gamma2_plot = [], []
    T_promedio = np.mean(T_exp) + 273.15  # Pasar a Kelvin para las ecuaciones
    
    for xi in x_calc_line:
        x_vec = [xi, 1.0 - xi]
        if model_type == "NRTL":
            g1, g2 = nrtl_gamma(x_vec, T_promedio, mejores_params[0], mejores_params[1], mejores_params[2])
        else:
            g1, g2 = uniquac_gamma(x_vec, T_promedio, mejores_params[0], mejores_params[1])
        gamma1_plot.append(g1)
        gamma2_plot.append(g2)

    ax2.plot(x_calc_line, gamma1_plot, color='#8e44ad', lw=2.5, label='$\gamma_1$ (Etanol)')
    ax2.plot(x_calc_line, gamma2_plot, color='#27ae60', lw=2.5, label='$\gamma_2$ (Agua)')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Comportamiento Ideal')
    
    ax2.set_title("Coeficientes de Actividad ($\gamma$) - Desviación del Ideal", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Fracción molar en líquido ($x_1$)", fontsize=10)
    ax2.set_ylabel("Coeficiente de Actividad ($\gamma$)", fontsize=10)
    ax2.legend(frameon=True, facecolor='white')

    # =========================================================================
    # 3. SUPERFICIE DE ERROR EN 3D + ENJAMBRE DE PARTICULAS
    # =========================================================================
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    
    # Malla de parámetros alrededor de los óptimos calculados por el algoritmo
    p1_center, p2_center = mejores_params[0], mejores_params[1]
    p1_range = np.linspace(max(-1000, p1_center - 500), min(2500, p1_center + 500), 40)
    p2_range = np.linspace(max(-1000, p2_center - 500), min(2500, p2_center + 500), 40)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Mapear la superficie de la Función Objetivo
    Z_of = np.zeros_like(P1)
    for r in range(P1.shape[0]):
        for c in range(P1.shape[1]):
            if model_type == "NRTL":
                test_params = [P1[r, c], P2[r, c], mejores_params[2]]
            else:
                test_params = [P1[r, c], P2[r, c]]
            Z_of[r, c] = objective_function(test_params, x_exp, y_exp, P_total, model_type)

    # Dibujar la topografía 3D difuminada
    surf = ax3.plot_surface(P1, P2, Z_of, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Proyectar curvas de contorno en el fondo de la gráfica (efecto mapa de relieve)
    ax3.contour(P1, P2, Z_of, zdir='z', offset=np.min(Z_of), cmap='viridis', alpha=0.7)

    # --- SIMULACIÓN DEL ENJAMBRE DE PARTÍCULAS INTERACTUANDO ---
    np.random.seed(42)
    n_visual_parts = 40
    p1_parts = np.random.uniform(p1_center - 400, p1_center + 400, n_visual_parts)
    p2_parts = np.random.uniform(p2_center - 400, p2_center + 400, n_visual_parts)
    
    # Acercar matemáticamente las partículas al óptimo (simulando los factores c1 y c2 del paper)
    p1_parts = p1_parts * 0.35 + p1_center * 0.65
    p2_parts = p2_parts * 0.35 + p2_center * 0.65
    
    z_parts = []
    for p1_p, p2_p in zip(p1_parts, p2_parts):
        if model_type == "NRTL":
            v_p = [p1_p, p2_p, mejores_params[2]]
        else:
            v_p = [p1_p, p2_p]
        z_parts.append(objective_function(v_p, x_exp, y_exp, P_total, model_type))

    # Esferas rojas: Las partículas buscando el mínimo global
    ax3.scatter(p1_parts, p2_parts, z_parts, color='red', s=40, edgecolor='black', 
                label='Partículas del enjambre (PSO)', alpha=0.8, zorder=10)
    
    # Gran estrella dorada: El g_best óptimo final encontrado
    z_best = objective_function(mejores_params, x_exp, y_exp, P_total, model_type)
    ax3.scatter(p1_center, p2_center, z_best, color='gold', s=250, marker='*', 
                edgecolor='black', label='Mínimo Global Óptimo ($g_{best}$)', alpha=1.0, zorder=15)

    # Etiquetas de ingeniería
    label_ij = "$B_{ij}$ (cal/mol)" if model_type == "NRTL" else "$U_{ij}$ (cal/mol)"
    label_ji = "$B_{ji}$ (cal/mol)" if model_type == "NRTL" else "$U_{ji}$ (cal/mol)"
    ax3.set_xlabel(label_ij, fontsize=10, labelpad=8)
    ax3.set_ylabel(label_ji, fontsize=10, labelpad=8)
    ax3.set_zlabel("Función Objetivo (Error)", fontsize=10, labelpad=8)
    ax3.set_title("Mapeo de la Superficie de Error y Convergencia de Partículas", fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right')
    
    # Ajustar ángulo tridimensional dinámico
    ax3.view_init(elev=25, azim=-130)

    plt.tight_layout()
    print("\n Desplegando el panel de visualización gráfica en alta definición...")
    plt.show()