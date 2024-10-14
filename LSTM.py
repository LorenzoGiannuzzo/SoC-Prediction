import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from df_importer import test_1a, test_2a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Supponiamo che 'test_1a' e 'test_2a' siano già caricati con i dati giusti, come nel tuo esempio.
# Extration of Charge and Discharge HPPC DATA
HPPC_D = HPPC_Dis.iloc[3:, :]
HPPC_C = HPPC_Cha.iloc[3:, :]

# Merging Charge and Discharge OCV DATA
ocv_data = pd.concat([OCV_Dis, OCV_Cha], axis=0)

# Merging Charge and Discharge HPPC DATA
HPPC_D_renamed = HPPC_D.iloc[:, [1, 2]].copy()
HPPC_C_renamed = HPPC_C.iloc[:, [1, 2]].copy()

HPPC_D_renamed.columns = ['Col1', 'Col2']
HPPC_C_renamed.columns = ['Col1', 'Col2']

x_full_train = pd.concat([HPPC_D_renamed, HPPC_C_renamed], axis=0, ignore_index=True, sort=False)

HPPC_D_renamed = HPPC_D.iloc[:, [3]].copy()
HPPC_C_renamed = HPPC_C.iloc[:, [3]].copy()

HPPC_D_renamed.columns = ['Col1']
HPPC_C_renamed.columns = ['Col1']

X_train = np.array(test_2a.iloc[3:, [1, 2]], dtype=np.float32)
y_train = np.array(test_2a.iloc[3:, [3]], dtype=np.float32)

x_full_train = pd.DataFrame(X_train)
y_full_train = y_train

# Normalize features (Volt, Current)
mean_V = x_full_train.iloc[:, 0].mean()
mean_I = x_full_train.iloc[:, 1].mean()
std_V = x_full_train.iloc[:, 0].std()
std_I = x_full_train.iloc[:, 1].std()

x_full_train.iloc[:, 0] = (x_full_train.iloc[:, 0] - mean_V) / std_V
x_full_train.iloc[:, 1] = (x_full_train.iloc[:, 1] - mean_I) / std_I

X_test = np.array(test_1a.iloc[3:, [1, 2]], dtype=np.float32)
y_test = np.array(test_1a.iloc[3:, [3]], dtype=np.float32)

X_test[:, 0] = (X_test[:, 0] - mean_V) / std_V
X_test[:, 1] = (X_test[:, 1] - mean_I) / std_I

# FEATURES CREATION
x_full_train = np.array(x_full_train)
x_full_test = np.array(X_test)

# Aggiunta dello stato di carica precedente come feature
soc_previous_train = np.roll(y_train, shift=1)
soc_previous_train[0] = y_train[0]  # Il primo valore non ha un precedente, lo manteniamo uguale
x_full_train = np.column_stack((x_full_train, soc_previous_train))

soc_previous_test = np.roll(y_test, shift=1)
soc_previous_test[0] = y_test[0]
x_full_test = np.column_stack((x_full_test, soc_previous_test))

# CREATING 3D ARRAY FOR LSTM
sample_size_train = x_full_train.shape[0]
sample_size_test = x_full_test.shape[0]
time_steps = x_full_train.shape[1]
input_dimension = 1

# RESHAPING THE INPUT DATA
x_full_train_reshaped = x_full_train.reshape(sample_size_train, time_steps, input_dimension)
x_full_test_reshaped = x_full_test.reshape(sample_size_test, time_steps, input_dimension)

print("After reshape training data shape:\n", x_full_train_reshaped.shape)
print("1 Sample shape:\n", x_full_train_reshaped[0].shape)
print("An example sample:\n", x_full_train_reshaped[0])


# BUILDING LSTM MODEL
def build_lstm_model():
    model = keras.Sequential(name="model_lstm")

    # LSTM layer
    model.add(keras.layers.LSTM(64, activation='relu',
                                input_shape=(x_full_train_reshaped.shape[1], x_full_train_reshaped.shape[2])))
    model.add(keras.layers.Dropout(0.5))

    # Dense layers
    model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(1, name="Output"))  # Output layer for SoC

    optimizer = 'adam'
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


model_lstm = build_lstm_model()
model_lstm.summary()

# TRAINING THE LSTM MODEL
history = model_lstm.fit(np.array(x_full_train_reshaped, dtype=np.float32),
                         np.array(y_full_train, dtype=np.float32),
                         epochs=50, validation_split=0.2, verbose=1)

# RECURSIVE PREDICTION
y_pred_recursive = []
current_soc = y_test[0]  # Primo SoC noto

# Loop attraverso i timestep
for i in range(len(x_full_test_reshaped)):
    # Prepara l'input per il modello: features + SoC precedente
    input_data = x_full_test_reshaped[i].reshape(1, -1, 1)

    # Predici il SoC corrente
    predicted_soc = model_lstm.predict(input_data)

    # Salva la predizione
    y_pred_recursive.append(predicted_soc[0][0])

    # Aggiorna lo stato di carica per il prossimo timestep
    current_soc = predicted_soc[0][0]

# Calcola metriche
y_pred_recursive = np.array(y_pred_recursive)

mae = mean_absolute_error(y_test[:, 0], y_pred_recursive)
mse = mean_squared_error(y_test[:, 0], y_pred_recursive)
r2 = r2_score(y_test[:, 0], y_pred_recursive)

print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}')

# Plot dei risultati
x = np.arange(len(y_test))

plt.figure(figsize=(8, 6))
plt.plot(x, y_test, label='Real Values (y_test)', color='blue')
plt.plot(x, y_pred_recursive, label='Predicted Values (y_pred)', color='orange')

plt.xlabel('Step')
plt.ylabel('SoC')
plt.title('Comparison between Real and Predicted Values')
plt.legend()
plt.show()
