#!/usr/bin/env python3
import cv2
import numpy as np

# images = [i for i in range(45, 51)]

images = [380, 383, 390, 414, 416, 419, 430, 449, 451, 452, 453, 455, 458, 459, 460, 461, 462, 463, 894, 901, 1377, 1383, 1384, 1386, 1388, 1389, 1394, 1401, 1403, 1404, 1405, 1406, 1407,
          1408, 1411, 1412, 1420, 1421, 1437, 1440, 1441, 1442, 1443, 1444, 1445, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1458, 1459, 1460, 1461, 1462, 1463, 1499, 
          1500, 1506, 1508, 1530, 1564, 1565, 1567, 1568, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1607, 1608, 1619, 1620, 1644, 1646, 1647, 
          1648, 1649, 1650, 1739, 1740, 1741, 1742, 1743, 1744, 1745]

for i in images:
    img = cv2.imread("/home/abhishek/catkin_ws/tools/ds4/Right/img" + str(i) + ".png")


    blur = cv2.GaussianBlur(img,(3,3),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #Mask for green
    lower_thresh_g = np.array([40, 55, 75], np.uint8)
    upper_thresh_g = np.array([70, 255, 255], np.uint8)

    # #Mask for red
    lower_thresh_r = np.array([130, 30, 69], np.uint8) #[140, 35, 58]
    upper_thresh_r = np.array([192, 255, 255], np.uint8) #[170, 255, 255]

    green_mask = cv2.inRange(hsv, lower_thresh_g, upper_thresh_g)
    red_mask = cv2.inRange(hsv, lower_thresh_r, upper_thresh_r)

    contours_g,_ = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_r,_ = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areasG = [cv2.contourArea(c) for c in contours_g]
    areasR = [cv2.contourArea(c) for c in contours_r]

    sorted_areasG = np.sort(areasG)
    sorted_areasR = np.sort(areasR)

    if sorted_areasG[-1] >= 5:
        cnt_g=contours_g[areasG.index(sorted_areasG[-1])] #the biggest contour
    if sorted_areasR[-1] >= 5:
        cnt_r=contours_r[areasR.index(sorted_areasR[-1])]

    print(sorted_areasG[-1])
    print(sorted_areasR[-1])


    try:
        ellipseG = cv2.fitEllipse(cnt_g)
        ellipseR = cv2.fitEllipse(cnt_r)
        cv2.ellipse(img,ellipseG,(0,255,0),2)
        cv2.ellipse(img,ellipseR,(0,0,255),2)
        mg = cv2.moments(cnt_g)
        gX = int(mg["m10"]/mg["m00"])
        gY = int(mg["m01"]/mg["m00"])
        cv2.circle(img, (gX, gY), 3, (0, 255, 0), -1)
        print(gX, gY)

    except:
        print("green ellipse exception")

    try:
        mr = cv2.moments(cnt_r)
        rX = int(mr["m10"]/mr["m00"])
        rY = int(mr["m01"]/mr["m00"])
        cv2.circle(img, (rX, rY), 3, (0, 0, 255), -1)
        print(rX, rY)
    except:
        print("red ellipse exception")



    cv2.imshow(str(i), img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
