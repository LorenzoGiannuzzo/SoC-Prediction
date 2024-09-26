import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from df_importer import test_1a
from df_explorer import HPPC_Dis, HPPC_Cha, OCV_Cha, OCV_Dis

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


ocv_data = pd.concat([OCV_Dis,OCV_Cha], axis=0)
hppc_data = pd.concat([HPPC_Dis,HPPC_Cha],axis=0)


model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(128, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}')

# Use the model to predict SOC
new_features = new_features.reshape(-1, 128, 1)
soc_pred = model.predict(new_features)
print(f'Predicted SOC: {soc_pred[0][0]:.2f}')