import pandas as pd
import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from df_importer import test_1a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis

from keras.api.models import Sequential
from keras.api.layers import Conv1D
from keras.api.layers import MaxPooling1D, Flatten, Dense
from keras import Input

# HPPC_D, HPPC_C, test_1a, OCV_Dis, OCV_Cha presupposti come importati prima
HPPC_D = HPPC_Dis.iloc[3:, :]
HPPC_C = HPPC_Cha.iloc[3:, :]

ocv_data = pd.concat([OCV_Dis, OCV_Cha], axis=0)
hppc_data = pd.concat([HPPC_D.iloc[:, :3], HPPC_C.iloc[:, :3]], axis=0)

# Training and test set
X_train = np.array(HPPC_D.iloc[:, [1, 2]], dtype=np.float32)
y_train = np.array(HPPC_D.iloc[:, [3]], dtype=np.float32)
X_test = np.array(test_1a.iloc[3:, [1, 2]], dtype=np.float32)
y_test = np.array(test_1a.iloc[3:, [3]], dtype=np.float32)

length = len(X_train)

# Reshape the data to 3D for Conv1D layer
X_train = X_train.reshape((1, len(X_train), 2))
X_test = X_test.reshape((1, len(X_test), 2))

# Reshape the labels to match the shape required
y_train = y_train.reshape((1, len(y_train), 1))
y_test = y_test.reshape((1, len(y_test), 1))

# Build the model
model = Sequential()

# Reduced kernel size and added padding='same' to avoid reducing the output size
model.add(Conv1D(1, kernel_size=1000, activation='relu', input_shape=(length, 2), padding='same'))
model.add(Dense(1,activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=500, validation_data=(X_test, y_test),verbose=1)

# Evaluate the model

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test[0,:,0], y_pred[0,:,0])
mse = mean_squared_error(y_test[0,:,0], y_pred[0,:,0])
r2 = r2_score(y_test[0,:,0], y_pred[0,:,0])
print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}')

