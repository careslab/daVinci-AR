#!/usr/bin/env python3

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import rospy
from cv_bridge import CvBridge
import cv2
import os
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf.transformations import quaternion_matrix, quaternion_from_euler
import image_geometry
import numpy as np
import time

class TimeFilter:
    def __init__(self):

        rospy.init_node ("time_filter", anonymous = True)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.ig = image_geometry.StereoCameraModel()
        self.cam_info = {"left" : [], "right": []}

        self.left_K = np.array([[732.5006713, 0.0, 289.79898],
                                 [0.0, 752.4523315, 194.3993],
                                 [0.0, 0.0, 1.0]])
        
        self.left_dist = np.array([-0.28349692200, 0.92551624100, 0.029140794, -0.001387055, 4.56057552])

        self.video = cv2.VideoWriter('/home/abhishek/da-vinci-ar/test_25mm_10frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640,480))

        self.cam_updated_left = True
        self.cam_updated_right = True
        out = '/home/abhishek/catkin_ws/tools/optim_cal/ds4/'
        self.csv_filename = '/home/abhishek/catkin_ws/tools/cv-projection.csv'

        self.path_left = os.path.join(out, 'Left')
        self.path_right = os.path.join(out, 'Right')
        # os.mkdir(self.path_left)
        # os.mkdir(self.path_right)

        # self.file_left = open(self.csv_filename, 'w')
        # self.file_right = open('/home/abhishek/catkin_ws/tools/optim_cal/ds4/right.csv', 'w')

        # self.camInfor_sub = rospy.Subscriber("/fakecam_node/camera_info", CameraInfo, self.cam_cb)
        

        self.imageL_sub = message_filters.Subscriber('/image_raw_left', Image)
        self.imageR_sub = message_filters.Subscriber('/image_raw_right', Image)
        self.psm1_sub = message_filters.Subscriber('/dvrk/PSM1/position_cartesian_current', PoseStamped)
        self.psm2_sub = message_filters.Subscriber('/dvrk/PSM2/position_cartesian_current', PoseStamped)
        self.ecm_sub = message_filters.Subscriber('/dvrk/ECM/position_cartesian_current', PoseStamped)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.imageL_sub, self.imageR_sub, self.psm1_sub, self.psm2_sub, self.ecm_sub], 12, 0.01)
        self.ts.registerCallback(self.callback)
    
    def cam_cb(self, msg):

        if msg.header.frame_id == "/fake_cam_left_optical_link":
            self.cam_info["right"] = msg
            self.cam_updated_left = True
        elif msg.header.frame_id == "/fake_cam_right_optical_link":
            self.cam_info["left"] = msg
            self.cam_updated_right = True
        

    
    def pose2matrix(self, pose_msg):
        
        matrix = quaternion_matrix([
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z,
        pose_msg.pose.orientation.w
        ])

        matrix[0][3] = pose_msg.pose.position.x
        matrix[1][3] = pose_msg.pose.position.y
        matrix[2][3] = pose_msg.pose.position.z

        return np.asmatrix(matrix)
    
    def callback(self, imageL, imageR, PSM1, PSM2, ECM):
        print("processing")
        global i
        br = CvBridge()

        if self.cam_updated_left and self.cam_updated_right:
            # self.ig.fromCameraInfo(self.cam_info["right"], self.cam_info["left"])

            PSM1.header.frame_id = 'one_psm_base_link'
            PSM2.header.frame_id = 'two_psm_base_link'
            ECM.header.frame_id = 'ecm_roll_link'
            
           

            self.tfBuffer.can_transform('world', 'one_psm_base_link', rospy.Time(), rospy.Duration(1.0))
            self.tfBuffer.can_transform('world', 'two_psm_base_link', rospy.Time(), rospy.Duration(1.0))
            self.tfBuffer.can_transform('world', 'ecm_roll_link', rospy.Time(), rospy.Duration(1.0))


            PSM1_world = do_transform_pose(PSM1, self.tfBuffer.lookup_transform('world', 'one_psm_base_link', rospy.Time()))
            PSM2_world = do_transform_pose(PSM2, self.tfBuffer.lookup_transform('world', 'two_psm_base_link', rospy.Time()))
            ECM_world = do_transform_pose(ECM, self.tfBuffer.lookup_transform('world', 'ecm_roll_link', rospy.Time()))
            imgL = br.imgmsg_to_cv2(imageR)
            # imgR = self.br.imgmsg_to_cv2(imageR)

            imgL_name = self.path_left + '/img' + str(imageL.header.seq) + '.png'
            imgR_name = self.path_right + '/img' + str(imageR.header.seq) + '.png'

            P1_matrix = self.pose2matrix(PSM1_world)
            P2_matrix = self.pose2matrix(PSM2_world)
            E_matrix = self.pose2matrix(ECM_world)

            # print(P1_matrix)

            trans = [0, 0, 0]
            quat = quaternion_from_euler(0.0, 0.0, 1.57079632679, 'sxyz')
            # print(quat)
            r_matrix = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])
            r_matrix[:3, 3] = trans
            r_matrix = np.asmatrix(r_matrix)
            r_inv = np.linalg.inv(r_matrix)
            E_inv = np.linalg.inv(E_matrix)

            # p = [[0.6724, 0.0328, 0.04374, 6.5488],
            #      [-0.4874, 0.2112, 1.1373, -1.3631],
            #      [-0.03, -0.0293, 0.06361, -0.2333],
            #      [0, 0, 0, 1]]
            
            # p = np.asmatrix(p)

            t1 = (r_inv * E_inv * P1_matrix )
            t2 = (r_inv * E_inv * P2_matrix )

            point1 = t1[0:3,3]
            point1[0] = point1[0] + 0.0025
            # point1[1] = point1[1] + 0.015
            point1 = np.expand_dims(point1, axis=0)

            point2 = t2[0:3,3]
            point2[0] = point2[0] + 0.0025
            # point2[1] = point2[1] + 0.01
            point2 = np.expand_dims(point2, axis=0)

            # print(str(point1[0][0][0]))
            # print(str(point1[0][1][0]))
            # print(str(point1[0][2][0]))
            
            # t1f = (t1*p)[0:3, 3]
            # t2f = (t2*p)[0:3, 3]

            # t1f = p*t1
            # t2f = p*t2

            # l1, r1 = self.ig.project3dToPixel(t1f) # tool1 left and right pixel positions
            # l2, r2 = self.ig.project3dToPixel(t2f) # tool2 left and right pixel positions

            l1, _ = cv2.projectPoints(point1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), self.left_K, self.left_dist)
            l2, _ = cv2.projectPoints(point2, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), self.left_K, self.left_dist)

            l1 = tuple(np.squeeze(l1))
            l2 = tuple(np.squeeze(l2))

            # lu = int((418.9575*t1f[0,0] - 28.2715)/t1f[0,2] + 431.9974)
            # lv = int((144.9231*t1f[0,1] - 1.4528)/t1f[0,2] + 187.7345)

            # lu2 = int((418.9575*t2f[0,0] - 28.2715)/t2f[0,2] + 431.9974)
            # lv2 = int((144.9231*t2f[0,1] - 1.4528)/t2f[0,2] + 187.7345)

            # print(lu, lv)

            # ru = (747.48913574*t1f[0,0] - 5.39561)/t1f[0,2] + 228.160290738
            # rv = (756.40936279*t1f[0,1] + 0.0197)/t1f[0,2] + 155.81636139
            # print(l1)

            cv2.circle(imgL, (int(l1[0]), int(l1[1])), 3, (0, 0, 255), 2)
            cv2.circle(imgL, (int(l2[0]), int(l2[1])), 3, (0, 255, 0), 2)

            self.video.write(imgL)
            # cv2.imshow("img", imgL)
            # cv2.waitKey(1)

            # print("3dtoPixel")
            # print(f"left: {l1} right: {r1}")

            # print((lu, lv), (ru, rv))

            self.file_left.write(str(point1[0][0][0]) + ',' + str(point1[0][1][0]) + ',' + str(point1[0][2][0]) + ',' + 
                            str(point2[0][0][0]) + ',' + str(point2[0][1][0]) + ',' + str(point2[0][2][0]) + ',' + 
                            str(int(l1[0])) + ',' + str(int(l1[1])) + ',' + str(int(l2[0])) + ',' + str(int(l2[1])) + ',' + imgL_name + ',' + imgR_name + '\n')
            # self.file_left.write(str(t2[0,0]) + ',' + str(t2[0,1]) + ',' + str(t2[0,2]) + ',' + str(int(l2[0])) + ',' + str(int(l2[1])) + '\n')
            # self.file_left.write(str(t1[0,0]) + ',' + str(t1[0,1]) + ',' + str(t1[0,2]) + ',' + str(int(l1[0])) + ',' + str(int(l1[1])) + '\n')

            # self.file_right.write(str(t2[0,0]) + ',' + str(t2[0,1]) + ',' + str(t2[0,2]) + ',' + str(int(r2[0])) + ',' + str(int(r2[1])) + '\n')
            # self.file_right.write(str(t1[0,0]) + ',' + str(t1[0,1]) + ',' + str(t1[0,2]) + ',' + str(int(r1[0])) + ',' + str(int(r1[1])) + '\n')

            # self.file.write(str(PSM2_world.pose.position.x) + ',' + str(PSM2_world.pose.position.y) + ',' + str(PSM2_world.pose.position.z) + ',' + 
            #                 str(ECM_world.pose.position.x) + ',' + str(ECM_world.pose.position.y) + ',' + str(ECM_world.pose.position.z) + ',' + 
            #                 str(int(l2[0])) + ',' + str(int(l2[1])) + ',' + str(int(r2[0])) + ',' + str(int(r2[1])) + ',' + imgL_name + ',' + imgR_name + '\n')
            
            # self.file.write(str(PSM1_world.pose.position.x) + ',' + str(PSM1_world.pose.position.y) + ',' + str(PSM1_world.pose.position.z) + ',' + 
            #                 str(ECM_world.pose.position.x) + ',' + str(ECM_world.pose.position.y) + ',' + str(ECM_world.pose.position.z) + ',' + 
            #                 str(int(l1[0])) + ',' + str(int(l1[1])) + ',' + str(int(r1[0])) + ',' + str(int(r1[1])) + ',' + imgL_name + ',' + imgR_name + '\n')
        
            # self.file.write(str(PSM2.pose.position.x) + ',' + str(PSM2.pose.position.y) + ',' + str(PSM2.pose.position.z) + ',' + 
            #                 str(ECM.pose.position.x) + ',' + str(ECM.pose.position.y) + ',' + str(ECM.pose.position.z) + ',' + imgL_name + ',' + imgR_name + '\n')
            
            # self.file.write(str(PSM1.pose.position.x) + ',' + str(PSM1.pose.position.y) + ',' + str(PSM1.pose.position.z) + ',' + 
            #                 str(ECM.pose.position.x) + ',' + str(ECM.pose.position.y) + ',' + str(ECM.pose.position.z) + ',' + imgL_name + ',' + imgR_name + '\n')

if __name__ == "__main__":

    # rospy.init_node("timefilter", anonymous=False)
    tf = TimeFilter()
    rospy.spin()
    # tf.file_right.close()
    # tf.file_left.close()
    # cv2.destroyAllWindows()
    tf.video.release()     