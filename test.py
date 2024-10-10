import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from df_importer import test_1a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis

from keras.api.models import Sequential
from keras.api.layers import Conv1D
from keras.api.layers import MaxPooling1D, Flatten, Dense

HPPC_D = HPPC_Dis.iloc[3:,:]
HPPC_C = HPPC_Cha.iloc[3:,:]

ocv_data = pd.concat([OCV_Dis,OCV_Cha], axis=0)
hppc_data = pd.concat([HPPC_D.iloc[:, :3], HPPC_C.iloc[:, :3]], axis=0)

X_train = np.array(HPPC_D.iloc[:, [1, 2]], dtype=np.float32)
y_train = np.array(HPPC_D.iloc[:, [3]], dtype=np.float32)
X_test = np.array(test_1a.iloc[3:, [1, 2]], dtype=np.float32)
y_test = np.array(test_1a.iloc[3:, [3]], dtype=np.float32)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=[len(X_train), 2]))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}')

