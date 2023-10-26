import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from colorfilters import HSVFilter
import imutils


img = cv2.imread('../tools/ds26/Left/img2656.png')
blur = cv2.GaussianBlur(img, (3, 3), 0)
img = blur
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# window = HSVFilter(img)
# window.show()

# hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# self.lower_thresh_g = np.array([40, 55, 75], np.uint8)
# self.upper_thresh_g = np.array([70, 255, 255], np.uint8)
#
# self.lower_thresh_r = np.array([130, 30, 69], np.uint8)  # [140, 35, 58]
# self.upper_thresh_r = np.array([192, 255, 255], np.uint8)  # [170, 255, 255]

lower_thresh_g = np.array([55, 70, 75], np.uint8)
upper_thresh_g = np.array([90, 255, 255], np.uint8)

lower_thresh_r = np.array([130, 30, 69], np.uint8)  # [140, 35, 58] [130, 30, 69]
upper_thresh_r = np.array([192, 255, 255], np.uint8)  # [170, 255, 255] [192, 255, 255]

green_mask = cv2.inRange(img, lower_thresh_g, upper_thresh_g)
red_mask = cv2.inRange(img, lower_thresh_r, upper_thresh_r)

cv2.imshow('red_mask', red_mask)
cv2.imshow('green_mask', green_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
