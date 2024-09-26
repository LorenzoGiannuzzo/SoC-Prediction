import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from df_importer import HPPC_Dis, HPPC_Cha

# Funzione per approssimare la risposta di rilassamento (della tensione) con un'esponenziale
def voltage_relaxation(t, R1, C1, V_inf):
    return V_inf + (V_0 - V_inf) * np.exp(-t / (R1 * C1))

# Funzione per calcolare i parametri del modello Equivalent Circuit Model (ECM)
def calculate_ECM_parameters(time, current, voltage):
    # Identificazione degli impulsi (corrente non zero e valore assoluto maggiore di 100 o minore di -200)
    impulse_indices = np.where((np.abs(current) > 0))[0]

    # Suddividere in segmenti per ogni impulso
    impulse_segments = []
    start_idx = impulse_indices[0]

    for i in range(1, len(impulse_indices)):
        # Se la corrente torna a zero o cambia segno, considera un nuovo impulso
        if current[impulse_indices[i]] == 0 or (
                current[impulse_indices[i]] > 0 and current[impulse_indices[i - 1]] < 0) or (
                current[impulse_indices[i]] < 0 and current[impulse_indices[i - 1]] > 0):
            impulse_segments.append((start_idx, impulse_indices[i - 1]))
            start_idx = impulse_indices[i]

    # Aggiungi l'ultimo impulso
    impulse_segments.append((start_idx, impulse_indices[-1]))

    results = []  # Per memorizzare i risultati per ogni impulso

    for start, end in impulse_segments:
        # Calcolo della resistenza interna R0 (caduta istantanea di tensione durante l'impulso)
        delta_V = voltage[start] - voltage[start - 1]  # Caduta di tensione immediata
        delta_I = current[start]  # Corrente al momento dell'impulso

        R_0 = delta_V / delta_I
        print(f"Resistenza interna R_0 = {R_0:.4f} Ohm")

        # Approssimazione della curva di rilassamento per calcolare R1 e C1
        t_rest = time[np.where(current == 0)[0]] - time[np.where(current == 0)[0]][0]  # Tempo durante il riposo
        V_rest = voltage[np.where(current == 0)[0]]

        # Fit esponenziale per ottenere R1 e C1
        global V_0
        V_0 = voltage[end]  # Tensione alla fine dell'impulso
        popt, pcov = curve_fit(voltage_relaxation, t_rest, V_rest, p0=[0.01, 2000, V_rest[-1]])

        R_1, C_1, V_inf = popt
        print(f"Resistenza di polarizzazione R_1 = {R_1:.4f} Ohm")
        print(f"Capacità di polarizzazione C_1 = {C_1:.4f} Farad")

        # Memorizza i risultati
        results.append((R_0, R_1, C_1))

        # Plot della curva di rilassamento per verificare il fit
        plt.plot(t_rest, V_rest, 'o', label='Dati misurati')
        plt.plot(t_rest, voltage_relaxation(t_rest, *popt), '-', label='Fit esponenziale')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Tensione [V]')
        plt.title(f'Impatto Impulso {len(results)}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return results  # Restituisce i risultati per ogni impulso

# Combinazione dei DataFrame
HPPC_combined = pd.concat([HPPC_Dis.iloc[:, [0, 1, 2, 3]], HPPC_Cha.iloc[:, [0, 1, 2, 3]]], axis=0)

for i in range(len(HPPC_combined)):
    if pd.isna(HPPC_combined.iloc[i, 1]):  # Controlla se il valore è NaN
        HPPC_combined.iloc[i, 1] = HPPC_combined.iloc[i, 4]  # Sostituisci con il valore della colonna 5

# Estrai le colonne di interesse e converti in array NumPy
time = pd.to_numeric(HPPC_combined.iloc[:, 0], errors='coerce').dropna().to_numpy()
current = pd.to_numeric(HPPC_combined.iloc[:, 2], errors='coerce').dropna().to_numpy()
voltage = pd.to_numeric(HPPC_combined.iloc[:, 1], errors='coerce').dropna().to_numpy()

# Calcola i parametri ECM per ogni impulso
results = calculate_ECM_parameters(time, current, voltage)

# Stampa i risultati finali
for i, (R_0, R_1, C_1) in enumerate(results):
    print(f"Impatto {i + 1}: R_0 = {R_0:.4f} Ohm, R_1 = {R_1:.4f} Ohm, C_1 = {C_1:.4f} Farad")

