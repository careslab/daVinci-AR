import pandas as pd
import numpy as np

dataset = pd.read_csv("/home/abhishek/catkin_ws/tools/combined_data/dataset.csv")
# print(dataset.max(axis=0))
dataset[abs(dataset) < 0.0001] = 0
dataset['Lr'] = dataset['Lr'].div(639)
dataset['Rr'] = dataset['Rr'].div(639)
dataset['Lc'] = dataset['Lc'].div(479)
dataset['Rc'] = dataset['Rc'].div(479)



cs = ["ToolX", "ToolY", "ToolZ", "CamX", "CamY", "CamZ"]

df = dataset.copy()

for c in cs:
    df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    print(f'col: {c} - min :{df[c].min}; max: {df[c].max}')
    # df[c] = df[c] * 1000

# print(df.iloc[25, :])
df.to_csv("/home/abhishek/catkin_ws/tools/combined_data/scaled_dataset.csv", index=False)
