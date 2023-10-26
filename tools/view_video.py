import os

import cv2 as cv
import pandas as pd
import pathlib

# PSM1 = green = right
# PSM2 = red = left

n = 1
num = 28
x_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_x.csv", header=None, skiprows=1)
y_data = pd.read_csv(f"./ds{num}/{num}_psm{n}_y.csv", header=None, skiprows=1)

X = x_data.iloc[:, 1:].values
y = y_data.iloc[:len(X), 1:].values

green = (0, 255, 0)
red = (255, 0, 0)

count = 0
print((y[count, 0], y[count, 1]))
cam = "Left"
for root, dir, imgs_n in os.walk(f"./ds{num}/{cam}"):
    for img_n in imgs_n:
        img = cv.imread(f'./ds{num}/{cam}/{img_n}')
        img = cv.circle(img, (int(y[count, 0]), int(y[count, 1])), radius=1, color=green)
        cv.imshow('Images', img)
        count = count + 1
