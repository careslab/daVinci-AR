import cv2
import numpy as np
import pandas as pd

file = "/home/abhishek/catkin_ws/tools/dataset6/tool_img_pos_right.csv"

data = pd.read_csv(file, header=None)
green_data = ()

for i, row in data.iterrows():
    # if row[0] < 1010:
    #     continue
    if i%2 == 0:
        green_data = (row[3], row[4])
        continue
    
    img = cv2.imread(row[1])
    red_data = (row[3], row[4])

    cv2.circle(img, green_data, 3, (0, 255, 0), -1)
    cv2.circle(img, red_data, 3, (0, 0, 255), -1)

    cv2.imshow(str(row[0]), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
