#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CameraInfo

left_info = CameraInfo()
right_info = CameraInfo()

#set left camera properties
left_info.header.frame_id = "/fake_cam_left_optical_link"
left_info.height = 480
left_info.width = 640
left_info.distortion_model = 'plumb_bob'
left_info.D = [-0.28349692200, 0.92551624100, 0.029140794, -0.001387055, 4.56057552]
left_info.K = [732.5006713, 0.0, 289.79898, 0.0, 752.4523315, 194.3993, 0.0, 0.0, 1.0]
# left_info.K = [418.9575, 0.0, 431.9974, 0.0, 144.9231, 187.7345, 0.0, 0.0, 1.0]
left_info.R = [0.998949, -0.00458, -0.045599, 0.004671, 0.99987, 0.001881, 0.04559, -0.0002092, 0.998958]
left_info.P = [732.5006713, 0.0, 289.79898, 0.0, 0.0, 752.4523315, 194.3993, 0.0, 0.0, 0.0, 1.0, 0.0]
left_info.binning_x = 0
left_info.binning_y = 0
left_info.roi.x_offset = 0
left_info.roi.y_offset = 0
left_info.roi.height = 0
left_info.roi.width = 0
left_info.roi.do_rectify = False

#set right camera properties
right_info.header.frame_id = "/fake_cam_right_optical_link"
right_info.height = 480
right_info.width = 640
right_info.distortion_model = 'plumb_bob'
right_info.D = [-0.3194892, -0.561046464, 0.0178256538, 0.001082061, 2.2380591]
right_info.K = [747.48-913574, 0.0, 228.160290738, 0.0, 756.40936279, 155.81636139, 0.0, 0.0, 1.0]
right_info.R = [0.999495, -0.012037, -0.029407, 0.011979, 0.999926, -0.002163, 0.029431, 0.00181, 0.999565]
right_info.P = [747.48913574, 0.0, 228.160290738, -5.39561, 0.0, 756.40936279, 155.81636139, 0.0197, 0.0, 0.0, 1.0, 0.0]
right_info.binning_x = 0
right_info.binning_y = 0
right_info.roi.x_offset = 0
right_info.roi.y_offset = 0
right_info.roi.height = 0
right_info.roi.width = 0
right_info.roi.do_rectify = False

rospy.init_node("fakecam", anonymous=True)
pub = rospy.Publisher("/fakecam_node/camera_info", CameraInfo, queue_size=10)
r = rospy.Rate(10)

while not rospy.is_shutdown():
    pub.publish(left_info)
    pub.publish(right_info)
    r.sleep()


