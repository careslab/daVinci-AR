import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import datetime
import shutil

# Read the CSV file
# data = pd.read_csv('dummy_dataset.csv', header=None, skiprows=1)
data = pd.read_csv("/home/abhishek/catkin_ws/tools/ds4/scaled_dataset.csv", header=None, skiprows=1)

# print(data.dtypes)

# Split the data into inputs (X) and outputs (y)
X = data.iloc[:, :6].values
y = data.iloc[:, 6:].values

# print(X[0])
# print(y[0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# print(X_test[0].shape)
# print(X_train[0], y_train[0])
model = Sequential([
    Dense(1024, activation='relu', input_shape=(6,)),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(4)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = os.path.join("logs", "lr_0.0001_adam_200e_nes_bs128")

# if os.path.isdir('./logs_world/scaled_data_500e_lr3e4'):
#     shutil.rmtree('./logs_world/scaled_data_500e_lr3e4')
#     print("removed folder")

log_dir = os.path.join("logs", "clean_scaled_data4_500e_lr3e4")

# Create the TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# esCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=10,
                                                # restore_best_weights=True, verbose=2)

# Train the model with the TensorBoard callback
history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])
# history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test))

model.save('model_v1.h5')

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test MSE: {mse}')

pred = model.predict(X_test)
print(pred[4])
print(y_test[4])
pred = pred * [639, 479, 639, 479]
print(y_test[4] * [639, 479, 639, 479])
print(pred[4])

model.summary()

# plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

