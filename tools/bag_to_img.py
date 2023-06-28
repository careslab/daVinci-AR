from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import rosbag
import csv
import os

br = CvBridge()
bag = rosbag.Bag('/home/abhishek/catkin_ws/bag_files/dataset3.bag')
out = '/home/abhishek/catkin_ws/tools/dataset3/'
csv_filename = '/home/abhishek/catkin_ws/tools/dataset3/dataset3.csv'

path_left = os.path.join(out, 'Left')
path_right = os.path.join(out, 'Right')

os.mkdir(path_left)
os.mkdir(path_right)

with open(csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    left_c = 0
    right_c = 0
    for topic, msg, t in bag.read_messages(topics=['/image_raw_left', '/image_raw_right']):

        img = br.imgmsg_to_cv2(msg)
        if topic == '/image_raw_left':
            filename = path_left + '/img' + str(left_c) + '.png'
            cv2.imwrite(filename, img)
            csvwriter.writerow([topic, t, filename])
            left_c += 1
        
        elif topic == '/image_raw_right':
            filename = path_right + '/img' + str(right_c) + '.png'
            cv2.imwrite(filename, img)
            csvwriter.writerow([topic, t, filename])
            right_c += 1
        else:
            print(f"unknown topic: {topic}")

csvfile.close()
bag.close()

