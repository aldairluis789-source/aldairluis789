import numpy as np
from tabulate import tabulate
from models.equilibrium import bubble_point_solver
from optimizers.pso import pso_lazzus_optimizer

BOUNDS_NRTL = np.array([[-1000.0, 2500.0], [-1000.0, 2500.0], [0.2, 0.4]])
BOUNDS_UNIQUAC = np.array([[-1000.0, 2500.0], [-1000.0, 2500.0]])

def ejecutar_programa():
    print("="*65)
    print(" OPTIMIZACIÓN TERMODINÁMICA - ALGORITMO PSO (LAZZÚS 2010)")
    print("="*65)

    print("Seleccione el modelo a optimizar:\n 1. NRTL\n 2. UNIQUAC")
    opcion = input("Ingrese opción (1 o 2): ").strip()
    model_type = "NRTL" if opcion == "1" else "UNIQUAC"
    bounds = BOUNDS_NRTL if model_type == "NRTL" else BOUNDS_UNIQUAC

    print("\nDefina la presión del sistema:")
    presion_unidades = input("¿Desea ingresar la presión en [1] kPa o [2] bar?: ").strip()
    p_ingresada = float(input("Ingrese el valor numérico de la presión: "))
    P_total = p_ingresada if presion_unidades == "1" else p_ingresada * 100.0

    print("\n--- INGRESO DE DATOS EXPERIMENTALES DEL ELV ---")
    usar_ejemplo = input("¿Desea cargar los datos de prueba del paper (Etanol+Agua)? [S/N]: ").strip().lower()

    if usar_ejemplo == 's':
        x_exp = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        y_exp = np.array([0.42, 0.52, 0.57, 0.65, 0.74, 0.88])
        T_exp = np.array([92.5, 87.3, 84.2, 80.7, 79.1, 78.3])
        P_total = 101.325
        print(" -> Datos del paper cargados exitosamente (Presión = 101.325 kPa).")
    else:
        n_puntos = int(input("¿Cuántos puntos experimentales va a ingresar?: "))
        x_list, y_list, t_list = [], [], []
        print("Ingrese cada fila separada por espacios o comas: x1 y1 T(°C)")
        for idx in range(n_puntos):
            fila = input(f"Punto {idx+1}: ").replace(',', ' ').split()
            x_list.append(float(fila[0]))
            y_list.append(float(fila[1]))
            t_list.append(float(fila[2]))
        x_exp = np.array(x_list)
        y_exp = np.array(y_list)
        T_exp = np.array(t_list)

    mejores_params, error_of = pso_lazzus_optimizer(bounds, x_exp, y_exp, P_total, model_type)

    print("\nGenerando reporte estadístico final...")
    sum_error_T, sum_error_y = 0.0, 0.0
    N_D = len(x_exp)
    lineas_analisis = []

    for i in range(N_D):
        T_calc, y1_calc = bubble_point_solver(x_exp[i], P_total, mejores_params, model_type)
        err_T = abs(T_exp[i] - T_calc)
        err_y = abs(y_exp[i] - y1_calc)
        sum_error_T += err_T
        sum_error_y += err_y
        lineas_analisis.append([x_exp[i], T_exp[i], T_calc, err_T, y_exp[i], y1_calc, err_y])

    delta_T = sum_error_T / N_D
    delta_y = sum_error_y / N_D

    print("\n" + "="*85)
    print(f"                 INFORME ESTADÍSTICO FINAL DE OPTIMIZACIÓN ({model_type})")
    print("="*85)
    if model_type == "NRTL":
        print(f"  * B_ij : {mejores_params[0]:10.4f} cal/mol\n  * B_ji : {mejores_params[1]:10.4f} cal/mol\n  * alpha: {mejores_params[2]:10.4f}")
    else:
        print(f"  * U_ij : {mejores_params[0]:10.4f} cal/mol\n  * U_ji : {mejores_params[1]:10.4f} cal/mol")

    print(f"\nValor mínimo de la Función Objetivo Global: {error_of:.6f}")
    headers = ["x1_exp", "T_exp (°C)", "T_calc (°C)", "Error T", "y1_exp", "y1_calc", "Error y1"]
    print(tabulate(lineas_analisis, headers=headers, tablefmt="fancy_grid", floatfmt=[".3f", ".2f", ".2f", ".3f", ".3f", ".3f", ".4f"]))
    print(f"\n DESVIACIÓN ABSOLUTA MEDIA DE TEMPERATURA  (Delta T) : {delta_T:.4f} °C")
    print(f" DESVIACIÓN ABSOLUTA MEDIA DE FASE VAPOR   (Delta y1): {delta_y:.4f}\n" + "="*85)

def main():
    while True:
        ejecutar_programa()
        if input("\n¿Desea hacer otra prueba? [S/N]: ").strip().lower() != 's':
            print("\n¡Programa finalizado con éxito!")
            break

if __name__ == "__main__":
    main()