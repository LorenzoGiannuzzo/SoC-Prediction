import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from df_importer import test_1a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Extration of Charge and Discharge HPPC DATA

HPPC_D = HPPC_Dis.iloc[3:, :]
HPPC_C = HPPC_Cha.iloc[3:, :]

# Merging Charge and Discharge OCV DATA

ocv_data = pd.concat([OCV_Dis, OCV_Cha], axis=0)

# Merging Charge and DIscharge HPPC DATA

HPPC_D_renamed = HPPC_D.iloc[:, [1,2]].copy()
HPPC_C_renamed = HPPC_C.iloc[:, [1,2]].copy()

HPPC_D_renamed.columns = ['Col1', 'Col2']
HPPC_C_renamed.columns = ['Col1', 'Col2']

x_full_train = pd.concat([HPPC_D_renamed, HPPC_C_renamed], axis=0, ignore_index=True, sort=False)

HPPC_D_renamed = HPPC_D.iloc[:, [3]].copy()
HPPC_C_renamed = HPPC_C.iloc[:, [3]].copy()

HPPC_D_renamed.columns = ['Col1']
HPPC_C_renamed.columns = ['Col1']

y_full_train = pd.concat([HPPC_D_renamed, HPPC_C_renamed], axis=0, ignore_index=True, sort=False)

# DATA Normalization

mean_V = x_full_train.iloc[:,0].mean()
mean_I = x_full_train.iloc[:,1].mean()

std_V = x_full_train.iloc[:,0].std()
std_I = x_full_train.iloc[:,1].std()

x_full_train.iloc[:,0] = (x_full_train.iloc[:,0] - mean_V) / std_V
x_full_train.iloc[:,1] = (x_full_train.iloc[:,1] - mean_I) / std_I


# Organization of TRAINING and TESTING DATA

X_train = np.array(x_full_train, dtype=np.float32)              #np.array(HPPC_D.iloc[:, [1, 2]], dtype=np.float32)
y_train = np.array(y_full_train, dtype=np.float32)              #np.array(HPPC_D.iloc[:, [3]], dtype=np.float32)

X_test = np.array(test_1a.iloc[3:, [1, 2]], dtype=np.float32)
y_test = np.array(test_1a.iloc[3:, [3]], dtype=np.float32)

X_test[:,0] = (X_test[:,0] - mean_V) / std_V
X_test[:,1] = (X_test[:,1] - mean_I) / std_I

print("Training set: {}".format(X_train.shape))
print("Testing set: {}".format(X_test.shape))

# MODEL CREATION - Multi-Layer Perceptrons (MLP)

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1)
        ], name="MLP_model")

    optimizer = 'adam'

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

model = build_model()
model.summary()

history = model.fit(X_train,y_train, epochs=50, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test, verbose=1)
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

