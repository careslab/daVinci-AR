import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
from keras.layers import Input, Dense, Concatenate, Add, GRU, Dropout, LayerNormalization, BatchNormalization, \
    SpatialDropout1D, SimpleRNN
from sklearn.preprocessing import StandardScaler
import os
import datetime
from keras.callbacks import TensorBoard

num = 28
n = 2
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_x.csv", header=None, skiprows=1)
y_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_y.csv", header=None, skiprows=1)

x_data = x_data.iloc[:, 1:].values
y_data = y_data.iloc[:len(x_data), 1:].values

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=64)
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


def create_dataset(X, y, look_back=1):
    Xs, ys = [], []

    for i in range(len(X) - look_back):
        # For taking in ONLY past PSM AND ECM positions
        v = X[i:i + (look_back - 1)]
        Xs.append(v[:, :])
        ys.append(y[i + (look_back - 1)])

        # For taking in current and past ONLY PSM positions
        # v = X[i:i+look_back]
        # Xs.append(v[:, 3:])
        # ys.append(X[i+look_back, :3])

    return np.array(Xs), np.array(ys)


LOOK_BACK = 3  # For "n" previous timesteps, set (LOOK_BACK = n + 1)
X_train, y_train = create_dataset(X_train, y_train, LOOK_BACK)
X_val, y_val = create_dataset(X_val, y_val, LOOK_BACK)
X_test, y_test = create_dataset(X_test, y_test, LOOK_BACK)
print(f"{X_train.shape} {y_train.shape}")
print(X_train[0])
print(y_train[0])
#
units = 16


# def create_model():
#     inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
#     # inputs = SpatialDropout1D(0.2)(inputs)
#     layer0_rnn = GRU(units=units, activation='tanh', recurrent_activation='sigmoid',
#                      return_sequences=True, input_shape=[X_train.shape[1], X_train.shape[2]])(inputs)
#     layer0_rnn = BatchNormalization()(layer0_rnn)
#     dense0 = Dense(units=units, activation='relu')(inputs)
#     res = Add()([dense0, layer0_rnn])
#     layer_rnn = GRU(units=units, activation='tanh', recurrent_activation='sigmoid',
#                     return_sequences=False)(res)
#     dropout1 = Dropout(0.2)(layer_rnn)
#     # dense1 = Dense(units=1024, activation='relu')(dropout1)
#     # dense1_gru = Concatenate()([dense1, layer_rnn])
#     # dropout2 = Dropout(0.3)(dense1_gru)
#     # dense2 = Dense(units=1024, activation='relu')(dropout2)
#     # dense2_gru = Concatenate()([dense2, layer_rnn])
#     # dropout3 = Dropout(0.3)(dense2_gru)
#     # dense3 = Dense(units=1024, activation='relu')(dropout3)
#     # outputs = Dense(4)(dense3)
#     dense1 = Dense(units=1024, activation=keras.layers.LeakyReLU())(dropout1)
#     dense1_gru = Concatenate()([dense1, layer_rnn])
#     dropout2 = Dropout(0.2)(dense1_gru)
#     dense2 = Dense(units=1024, activation=keras.layers.LeakyReLU())(dropout2)
#     dense2_gru = Concatenate()([dense2, layer_rnn])
#     dropout3 = Dropout(0.2)(dense2_gru)
#     dense3 = Dense(units=1024, activation=keras.layers.LeakyReLU())(dropout3)
#     outputs = Dense(4)(dense3)
#     model_out = keras.Model(inputs=inputs, outputs=outputs)
#     model_out.compile(keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mse'])
#     return model_out
def create_model():
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    # inputs = SpatialDropout1D(0.1)(inputs)
    # layer0_rnn = GRU(units=units, activation='tanh', recurrent_activation='sigmoid',
    #                  return_sequences=False, input_shape=[X_train.shape[1], X_train.shape[2]])(inputs)
    layer0_rnn = SimpleRNN(units=units, activation='tanh', return_sequences=False,
                           input_shape=[X_train.shape[1], X_train.shape[2]])(inputs)
    layer0_rnn = BatchNormalization()(layer0_rnn)
    dense0 = Dense(units=units, activation='relu')(layer0_rnn)
    res = Add()([dense0, layer0_rnn])
    dense1 = Dense(units=512, activation='relu')(res)
    dropout2 = Dropout(0.1)(dense1)
    # dense2 = Dense(units=512, activation=keras.layers.LeakyReLU())(dropout2)
    # dropout3 = Dropout(0.2)(dense2)
    # dense3 = Dense(units=512, activation=keras.layers.LeakyReLU())(dropout3)
    outputs = Dense(4)(dropout2)
    model_out = keras.Model(inputs=inputs, outputs=outputs)
    model_out.compile(keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mse'])
    return model_out


# keras.layers.LeakyReLU()


model = create_model()
model.summary()
# keras.utils.plot_model(
#     model,
#     to_file="model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=200,
#     layer_range=None,
#     show_layer_activations=False,
#     show_trainable=False,
# )


# 'mean_squared_error'
# 'huber'
#
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_psm{n}_gru")
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
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback])
# history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test))

model.save(f'./ds{num}/model_gru_{num}_{n}.keras')
joblib.dump(x_scaler, f"./ds{num}/{n}_gru_x_scaler.gz")
joblib.dump(y_scaler, f"./ds{num}/{n}_gru_y_scaler.gz")
# # Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test MSE: {mse}')
#
pred = model.predict(X_test)
# print(pred[4])
print(y_scaler.inverse_transform(np.expand_dims(pred[4], axis=0)))
print(y_scaler.inverse_transform(np.expand_dims(y_test[4], axis=0)))
print(y_test[4])
# pred = pred * [639, 479, 639, 479]
# print(y_test[4] * [639, 479, 639, 479])
# print(pred[4])
#
model.summary()
