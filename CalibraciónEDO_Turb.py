import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def leerdatos():
    ruta_archivo = "C:/Users/juanj/OneDrive/Escritorio/Python3.9/CalibraciónModeloHidrológico/Turb_2019-2020.xlsx"
    df = pd.read_excel(ruta_archivo, sheet_name="Datos")
    
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df = df.dropna()
    return df

def modelo_turbidez(Datos):  # Fecha | Turbidez | Precipitación | Afluente | Volumen | Viento | Radiación
    Fecha = Datos.iloc[:, 0]
    Turb_real = Datos.iloc[:, 1].values
    Precip = Datos.iloc[:, 2].values
    Afluente = Datos.iloc[:, 3].values
    Volumen = Datos.iloc[:, 4].values
    Viento = Datos.iloc[:, 5].values

    def simular_turbidez(params):
        a, b, c, d, e, f = params  # 7 parámetros
        Turb_modelada = [Turb_real[0]]

        for t in range(1, len(Fecha)):
            dNTUdt = (a * Precip[t-1] +
                      b * Afluente[t-1] +
                      c * Viento[t-1] -
                      d * Volumen[t-1]/1000000 +
                      e * Turb_modelada[-1] +
                      f)
            NTU_siguiente = Turb_modelada[-1] + dNTUdt
            Turb_modelada.append(NTU_siguiente)

        return np.array(Turb_modelada)

    def error(params):
        Turb_sim = simular_turbidez(params)
        return np.mean((Turb_sim - Turb_real)**2)

    params_iniciales = [0.2, 0.2, 0.3, 0.4, 0.1, 0.4]
    resultado = minimize(error, params_iniciales, method='Nelder-Mead')
    params_optimos = resultado.x

    print("\nParámetros calibrados (a, b, c, d, e, f):", params_optimos)
    a, b, c, d, e, f = params_optimos
    Turb_ajustada = simular_turbidez(params_optimos)

    # Calcular R2
    SS_res = np.sum((Turb_real - Turb_ajustada)**2)
    SS_tot = np.sum((Turb_real - np.mean(Turb_real))**2)
    r2 = 1 - (SS_res / SS_tot)
    MSE = np.mean((Turb_real - Turb_ajustada)**2)
    print(f"\nCoeficiente de determinación (MSE) del modelo: {MSE:.4f}")

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(Fecha, Turb_real, label='Turbidez Real (NTU)', color='blue')
    plt.plot(Fecha, Turb_ajustada, label='Turbidez Modelada (NTU)', color='red', linestyle='--')
    plt.xlabel("Fecha")
    plt.ylabel("Turbidez [NTU]")
    plt.legend()
    plt.grid(True)
    plt.title(f"Comparación entre Turbidez Real y Modelada\n$R^2$ = {r2:.4f}, MSE = {MSE:.4f}")
    plt.tight_layout()
    plt.show()

    data = pd.DataFrame({'Turbidez_real': Turb_real, 'Turbidez_modelada': Turb_ajustada}, index=Fecha)
    return data, a, b, c, d, e, f, r2

# ----------- EJECUTAR ------------
Datos = leerdatos()
Resultados, a, b, c, d, e, f, r2 = modelo_turbidez(Datos)

print("\nEcuación diferencial calibrada para Turbidez:")
print(f"d(NTU)/dt = {a:.5f}*Precipitación + {b:.5f}*Afluente + {c:.5f}*Viento "
      f"- {d:.5f}*(Volumen/1000000) + {e:.5f}*NTU[t-1]+ {f:.5f}")
