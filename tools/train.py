#! /usr/bin/env python3
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
# from keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import datetime
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib

x_scaler = StandardScaler()
y_scaler = StandardScaler()
# scaler = MinMaxScaler()

num = 28  # dataset number
n = 1  # psm number

# Read the CSV file
# data = pd.read_csv('dummy_dataset.csv', header=None, skiprows=1)
x_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_x.csv", header=None, skiprows=1)
y_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_y.csv", header=None, skiprows=1)

# x_data = x_data.dropna()
# y_data = y_data.dropna()
X = x_data.iloc[:, 1:].values
y = y_data.iloc[:len(X), 1:].values
print(X.shape)
print(y.shape)

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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

# print(X_test[0])
# print(X_test[0].shape)
print(X_train.shape, y_train.shape)
model = Sequential([
    Dense(1024, activation='relu', input_shape=(6,), ),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dropout(0.1),
    Dense(1024, activation='relu'),
    Dense(4)
])

optimizer = keras.optimizers.Adam(learning_rate=1e-4)
# 'mean_squared_error'
# 'huber'
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
#
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_psm{n}")
# log_dir = os.path.join("logs", "lr_0.0001_adam_200e_nes_bs128")

# if os.path.isdir('./logs_world/scaled_data_500e_lr3e4'):
#     shutil.rmtree('./logs_world/scaled_data_500e_lr3e4')
#     print("removed folder")

# log_dir = os.path.join("logs", "clean_scaled_data4_500e_lr3e4")

# # Create the TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#
esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=10,
                                           restore_best_weights=True, verbose=2)

# # Train the model with the TensorBoard callback
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])
# history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test))

model.save(f'./ds{num}/model_v1_{num}_{n}.keras')
joblib.dump(x_scaler, f"./ds{num}/{n}_x_scaler.gz")
joblib.dump(y_scaler, f"./ds{num}/{n}_y_scaler.gz")
# # Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test MSE: {mse}')
#
pred = model.predict(X_test)

print(y_scaler.inverse_transform(np.expand_dims(pred[4], axis=0)))
print(y_scaler.inverse_transform(np.expand_dims(y_test[4], axis=0)))
print(y_test[4])
# pred = pred * [639, 479, 639, 479]
# print(y_test[4] * [639, 479, 639, 479])
# print(pred[4])

model.summary()

# # plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
