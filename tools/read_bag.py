#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import rosbag
import csv

# rospack = rospkg.RosPack()
# pkg_path = rospack.get_path('tools')
br = CvBridge()
bag = rosbag.Bag('/home/abhishek/catkin_ws/bag_files/test_images.bag')

csv_filename = '/home/abhishek/catkin_ws/tools/test.csv'

with open(csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for topic, msg, t in bag.read_messages(topics=['/image_raw_left', '/image_raw_right']):
        img = br.imgmsg_to_cv2(msg)
        blur = cv2.GaussianBlur(img,(3,3),0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        #Mask for green
        lower_thresh_g = np.array([40, 55, 75], np.uint8)
        upper_thresh_g = np.array([70, 255, 255], np.uint8)

        #Mask for red
        lower_thresh_r = np.array([130, 30, 69], np.uint8)
        upper_thresh_r = np.array([170, 255, 255], np.uint8)

        green_mask = cv2.inRange(hsv, lower_thresh_g, upper_thresh_g)
        red_mask = cv2.inRange(hsv, lower_thresh_r, upper_thresh_r)

        contours_g,_ = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_r,_ = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        areasG = [cv2.contourArea(c) for c in contours_g]
        areasR = [cv2.contourArea(c) for c in contours_r]

        sorted_areasG = np.sort(areasG)
        sorted_areasR = np.sort(areasR)

        print(sorted_areasG)
        print(sorted_areasR)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        


        try:
            # ellipseG = cv2.fitEllipse(cnt_g)
            # ellipseR = cv2.fitEllipse(cnt_r)
            # cv2.ellipse(img,ellipseG,(0,255,0),2)
            # cv2.ellipse(img,ellipseR,(0,0,255),2)
            cnt_g=contours_g[areasG.index(sorted_areasG[-1])] #the biggest contour
            mg = cv2.moments(cnt_g)
            gX = int(mg["m10"]/mg["m00"])
            gY = int(mg["m01"]/mg["m00"])
            csvwriter.writerow([topic, 'green', gX, gY])

        except:
            print("error processing image")

        try:
            cnt_r=contours_r[areasR.index(sorted_areasR[-1])]
            mr = cv2.moments(cnt_r)
            rX = int(mr["m10"]/mr["m00"])
            rY = int(mr["m01"]/mr["m00"])
            csvwriter.writerow([topic, 'red', rX, rY])
        except:
            print("error processing image red")
    
        
csvfile.close()
bag.close()
