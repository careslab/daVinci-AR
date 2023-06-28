import numpy as np
import pandas as pd

inputs = np.random.rand(40, 6)

# Define nonlinear functions for each output

def f1(x):
    return np.sin(x[:, 0]) + np.cos(x[:, 1]) ** 2 + np.tanh(x[:, 2])

def f2(x):
    return np.exp(-x[:, 3]) + x[:, 4] ** 2 + np.log(x[:, 5] + 1)

def f3(x):
    return np.sqrt(np.abs(x[:, 0] * x[:, 1] - x[:, 2])) + np.arctan(x[:, 3])

def f4(x):
    return np.cbrt(np.abs(x[:, 4] - x[:, 5])) + np.sinh(x[:, 0])

# Calculate outputs using nonlinear functions

outputs = np.column_stack((f1(inputs), f2(inputs), f3(inputs), f4(inputs)))
data = np.column_stack((inputs, outputs))

df = pd.DataFrame(data, columns = ['in1','in2','in3','in4','in5','in6','out1','out2','out3', 'out4'])

df.to_csv("dummy_dataset.csv", index=False)