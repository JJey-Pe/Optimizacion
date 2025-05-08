import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def leerdatos():
    ruta_archivo = "C:/Users/juanj/OneDrive/Escritorio/Python3.9/CalibraciónModeloHidrológico/Temp_2019-2020.xlsx"
    df = pd.read_excel(ruta_archivo, sheet_name="2019-2020")

    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df = df.dropna(subset=df.columns[:6])
    df = df[(df[df.columns[1]] != 0) & 
            (df[df.columns[2]] != 0) & 
            (df[df.columns[3]] != 0) & 
            (df[df.columns[4]] != 0) & 
            (df[df.columns[5]] != 0) & 
            (df[df.columns[6]] != 0)]
    return df

def modelo_tempagua(Datos):  # Fecha | TempAgua | Radiación | Viento | TempAmb | Volumen | Presión
    Fecha = Datos.iloc[:, 0]
    TempAgua_real = Datos.iloc[:, 1].values
    Radiacion = Datos.iloc[:, 2].values
    Viento = Datos.iloc[:, 3].values
    TempAmb = Datos.iloc[:, 4].values
    Volumen = Datos.iloc[:, 5].values
    Presion = Datos.iloc[:, 6].values

    def simular_tempagua(params):
        a, b, c, d, e, f, g = params
        Temp_modelada = [TempAgua_real[0]]

        for t in range(1, len(Fecha)):
            dTdt = (a * TempAmb[t-1] +
                    b * Radiacion[t-1] +
                    c * Viento[t-1] +
                    d * Temp_modelada[-1] +
                    e * Volumen[t-1]/1000000 +
                    f * (Presion[t-1] / 1013) +
                    g)
            Temp_siguiente = Temp_modelada[-1] + dTdt
            Temp_modelada.append(Temp_siguiente)

        return np.array(Temp_modelada)

    def error(params):
        Temp_sim = simular_tempagua(params)
        return np.mean((Temp_sim - TempAgua_real) ** 2)

    params_iniciales = [0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0]
    resultado = minimize(error, params_iniciales, method='Nelder-Mead')
    params_optimos = resultado.x

    a, b, c, d, e, f, g = params_optimos
    print("\nParámetros calibrados (a, b, c, d, e, f, g):", params_optimos)

    Temp_ajustada = simular_tempagua(params_optimos)
    r2 = r2_score(TempAgua_real, Temp_ajustada)
    MSE = np.mean((TempAgua_real - Temp_ajustada)**2)
    print(f"Coeficiente de determinación MSE: {MSE:.5f}")

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(Fecha, TempAgua_real, label='Temperatura Agua Real', color='blue')
    plt.plot(Fecha, Temp_ajustada, label='Temperatura Agua Modelada', color='red', linestyle='--')
    plt.xlabel("Fecha")
    plt.ylabel("Temperatura del Agua [°C]")
    plt.title(f"Comparación entre Temperatura Real y Modelada\n$R^2$ = {r2:.4f}, MSE = {MSE:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    data = pd.DataFrame({'TempAgua_real': TempAgua_real, 'TempAgua_modelada': Temp_ajustada}, index=Fecha)
    return data, a, b, c, d, e, f, g, r2

Datos = leerdatos()
Resultados, a, b, c, d, e, f, g, r2 = modelo_tempagua(Datos)

print("\nEcuación diferencial calibrada para Temperatura del Agua:")
print(f"d(T_agua)/dt = {a:.5f}*T_ambiente + {b:.5f}*Radiación + "
      f"{c:.5f}*Viento + {d:.5f}*T_agua[t-1] + {e:.5f}*Volumen/1000000 + {f:.5f}*(Presión/1013) + {g:.5f}")
