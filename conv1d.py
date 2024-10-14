import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from df_importer import test_1a, test_2a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Extration of Charge and Discharge HPPC DATA

HPPC_D = HPPC_Dis.iloc[3:, :]
HPPC_C = HPPC_Cha.iloc[3:, :]

# Merging Charge and Discharge OCV DATA

ocv_data = pd.concat([OCV_Dis, OCV_Cha], axis=0)

# Merging Charge and Discharge HPPC DATA

HPPC_D_renamed = HPPC_D.iloc[:, [1,2]].copy()
HPPC_C_renamed = HPPC_C.iloc[:, [1,2]].copy()

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

#y_full_train = pd.concat([HPPC_D_renamed, HPPC_C_renamed], axis=0, ignore_index=True, sort=False)

mean_V = x_full_train.iloc[:,0].mean()
mean_I = x_full_train.iloc[:,1].mean()

std_V = x_full_train.iloc[:,0].std()
std_I = x_full_train.iloc[:,1].std()

x_full_train.iloc[:,0] = (x_full_train.iloc[:,0] - mean_V) / std_V
x_full_train.iloc[:,1] = (x_full_train.iloc[:,1] - mean_I) / std_I

X_test = np.array(test_1a.iloc[3:, [1, 2]], dtype=np.float32)
y_test = np.array(test_1a.iloc[3:, [3]], dtype=np.float32)

X_test[:,0] = (X_test[:,0] - mean_V) / std_V
X_test[:,1] = (X_test[:,1] - mean_I) / std_I

# FEATURES CREATION

x_full_train = np.array(x_full_train)

col2 = x_full_train[:, 0] * x_full_train[:, 1]
x_full_train = np.column_stack((x_full_train, col2))

diff_values_0 = np.diff(x_full_train[:, 0], prepend=0)
x_full_train = np.column_stack((x_full_train, diff_values_0))

diff_values_1 = np.diff(x_full_train[:, 1], prepend=0)
x_full_train = np.column_stack((x_full_train, diff_values_1))

# CREATING 3D ARRAY FOR CONV1D

sample_size = x_full_train.shape[0]
time_steps = x_full_train.shape[1]
input_dimension = 1

# RESHAPING THE INPUT DATA

x_full_train = np.array(x_full_train, dtype=np.float32)


x_full_train_reshaped = x_full_train.reshape(sample_size, time_steps, input_dimension)

# FEATURES CREATION

x_full_test = np.array(X_test)

col2 = x_full_test[:, 0] * x_full_test[:, 1]
x_full_test = np.column_stack((x_full_test, col2))

diff_values_0 = np.diff(x_full_test[:, 0], prepend=0)
x_full_test = np.column_stack((x_full_test, diff_values_0))

diff_values_1 = np.diff(x_full_test[:, 1], prepend=0)
x_full_test = np.column_stack((x_full_test, diff_values_1))

# CREATING 3D ARRAY FOR CONV1D

sample_size = x_full_test.shape[0]
time_steps = x_full_test.shape[1]
input_dimension = 1

# RESHAPING THE INPUT DATA

x_full_test = np.array(x_full_test, dtype=np.float32)

x_full_test_reshaped = x_full_test.reshape(sample_size, time_steps, input_dimension)

print("After reshape training data shape:\n",x_full_train_reshaped.shape)
print("1 Sample shape:\n", x_full_train_reshaped[0].shape)
print("An example sample:\n", x_full_train_reshaped[0])

# BUILDING CONV1D MODEL

def build_conv1D_model():

    n_timesteps = x_full_train_reshaped.shape[1]
    n_features = x_full_train_reshaped.shape[2]
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps,n_features)))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.MaxPooling1D(pool_size=1,name="MaxPooling1D"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(n_features, name="Dense_2"))

    optimizer = 'adam'

    model.compile(loss='mse',optimizer=optimizer, metrics=['mae'])

    return model


model_conv1D = build_conv1D_model()
model_conv1D.summary()

history = model_conv1D.fit(np.array(x_full_train_reshaped,dtype=np.float32), np.array(y_full_train, dtype=np.float32), epochs=50, validation_split=0.2, verbose=1)

y_pred = model_conv1D.predict(x_full_test_reshaped, verbose=1)
mae = mean_absolute_error(y_test[:,0], y_pred[:,0])
mse = mean_squared_error(y_test[:,0], y_pred[:,0])
r2 = r2_score(y_test[:,0], y_pred[:,0])
print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}')

x = np.arange(len(y_test))

plt.figure(figsize=(8, 6))
plt.plot(x, y_test, label='Real Values (y_test)', color='blue')
plt.plot(x, y_pred, label='Predicted Values (y_pred)', color='orange')

plt.xlabel('Step')
plt.ylabel('SoC')

plt.title('Comparison between Real and Predicted Values')
plt.legend()

plt.show()
