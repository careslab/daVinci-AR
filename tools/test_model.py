import keras.layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
# from keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import datetime
import shutil
import joblib

num = 28
n = 2
x_scaler = joblib.load(f"./ds{num}/{n}_x_scaler.gz")
y_scaler = joblib.load(f"./ds{num}/{n}_y_scaler.gz")

x_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_x.csv", header=None, skiprows=1)
y_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_y.csv", header=None, skiprows=1)

# x_data = x_data.dropna()
# y_data = y_data.dropna()
X = x_data.iloc[:, 1:].values
y = y_data.iloc[:len(X), 1:].values
# 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=64)

x_scaler.fit(X_train)
X_train = x_scaler.transform(X_train)
X_test = x_scaler.transform(X_test)
X_val = x_scaler.transform(X_val)
y_train_us = np.copy(y_train)
y_test_us = np.copy(y_test)
y_val_us = np.copy(y_val)

y_scaler.fit(y_train)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)
y_val = y_scaler.transform(y_val)

model = keras.models.load_model(f'./ds{num}/model_v1_{num}_{n}.keras')

# # Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test MSE: {mse}')
#
pred = model.predict(X_test)

round_pred = np.around(y_scaler.inverse_transform(pred), decimals=0)
loss = keras.losses.mean_absolute_error(y_test_us, round_pred)
print(y_test_us.shape, " ", round_pred.shape)
print(np.sum(loss) / len(loss))
print(round_pred[4])
# print(y_scaler.inverse_transform(np.expand_dims(y_test[4], axis=0)))
print(y_test_us[4])
# pred = pred * [639, 479, 639, 479]
# print(y_test[4] * [639, 479, 639, 479])
# print(pred[4])
#
model.summary()
