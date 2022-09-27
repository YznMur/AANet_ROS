#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import enum
import os
import sys
from cv2 import Mat
# from cv2 import transform
import numpy as np
import torch
import cv2
import tf2_ros
# from geometry_msgs.msg import TransformStamped
# import nav_msgs.msg
# import geometry_msgs.msg
from nav_msgs.msg import Odometry
# from math import degrees
# import tf_conversions
import math
from aanet_stereo_matching import AANetStereoMatcher, AANetStereoMatcherConfig
from utility import Utility as utils
from ros_node_base import RosNodeBase
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from transformations import quaternion_matrix

class ObjectTFConverter:
    def __init__(self):
        self.fixed_frame = rospy.get_param("~fixed_frame", "camera_gray_left")
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.prev_transform = None  
        # self.imu2velo = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01], [-7.854027e-04, 9.998898e-01, -1.482298e-02, 3.195559e-01], [2.024406e-03, 1.482454e-02, 9.998881e-01, -7.997231e-01], [0, 0, 0, 1]])
        # self.velo2cam = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03], [-2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02], [9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01], [0, 0, 0, 1]])
        # self.camtocam0 = np.array([[1.000000e+00, 0.000000e+00, 0.000000e+00, -1.850372e-17], [0.000000e+00, 1.000000e+00, 0.000000e+00, 6.938894e-17], [0.000000e+00, 0.000000e+00, 1.000000e+00, -7.401487e-17], [0, 0, 0, 1]])
        # self.camtocam1 = np.array([[9.993424e-01, 1.830363e-02, -3.129928e-02, -5.370000e-01],[-1.856768e-02, 9.997943e-01, -8.166432e-03, 5.591661e-03 ], [3.114337e-02, 8.742218e-03, 9.994767e-01, -1.200541e-02], [0, 0, 0, 1]])
        self.imu2velo_rot = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03], [-7.854027e-04, 9.998898e-01, -1.482298e-02], [2.024406e-03, 1.482454e-02, 9.998881e-01]])
        self.velo2cam_rot = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05], [-2.254837e-03, -4.184312e-05, -9.999975e-01], [9.999728e-01, 7.027479e-03, -2.255075e-03]])
        
    def _get_transform(self, frame_id, stamp):  
        try:
            transform = self.buffer.lookup_transform(
                self.fixed_frame, frame_id, stamp, rospy.Duration(0.001)
            )
            self.prev_transform = transform
            return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            # rospy.logwarn("Transformation waittime limit reached, using previous transformation")
            return self.prev_transform

    # def to_positive_angle(self, th):
    #     while True:
    #         if th < 0:
    #             th += 360
    #         if th > 0:
    #             ans = th % 360
    #             return ans
    #             break
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
        
    def eul_from_quat(self, x, y, z, w):
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        # print(roll_x*180/math.pi,"###",pitch_y*180/math.pi,"###", yaw_z*180/math.pi,"///////")
        return roll_x, pitch_y, yaw_z

                # Trans = np.matmul( self.velo2cam_rot,self.imu2velo_rot)

    # odom_msg.pose.pose.orientation.y,
    # odom_msg.pose.pose.orientation.z * -1, 
    # odom_msg.pose.pose.orientation.x * -1,
    # odom_msg.pose.pose.orientation.w, 
    # 
    #    
    # odom_msg.pose.pose.position.y * -1,
    # odom_msg.pose.pose.position.z * -1,
    # odom_msg.pose.pose.position.x,
    def transform_pose(self, outputs, header , odom_msg):
        # t = self._get_transform(header.frame_id, header.stamp)

        # x = odom_msg.pose.pose.orientation.x
        # y = odom_msg.pose.pose.orientation.y 
        # z = odom_msg.pose.pose.orientation.z 
        # w = odom_msg.pose.pose.orientation.w
        # self.eul_from_quat(x, y, z, w) 
        # roll_x_th, pitch_y_th, yaw_z_th = self.eul_from_quat(x, y, z, w) 

        
        # if (roll_x_th <= 0 and yaw_z_th <= 0):
        #     roll_x_th = (-roll_x_th)
        #     pitch_y_th = (-pitch_y_th)
        #     yaw_z_th = (-yaw_z_th)
            

        # if yaw_z_th <= -np.pi:
        #     yaw_z_th = (-yaw_z_th)

        
            # z = np.cos(th)
            # w = np.sin(th) * -1
        # quaternion = self.get_quaternion_from_euler(roll_x_th, pitch_y_th, yaw_z_th)
        # print (quaternion)
            # z = odom_msg.pose.pose.orientation.x * -1
            # w = odom_msg.pose.pose.orientation.w
        # print(odom_msg.pose.pose.orientation.x,"##",odom_msg.pose.pose.orientation.y,"##", odom_msg.pose.pose.orientation.z,"##",odom_msg.pose.pose.orientation.w,"##" )
        M_num = quaternion_matrix(
            np.array(
            [
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z * -1, 
                odom_msg.pose.pose.orientation.x * -1,
                odom_msg.pose.pose.orientation.w,
            ]
            ),
            iftorch=False,
        )

        p_num = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ],
        )
    
        # M_num = M.detach().cpu().numpy()
        # p_num = p.detach().cpu().numpy()
        # Mat = np.c_[ M_num[:4, :3], p_num ]
        # M0 = quaternion_matrix(
        #     torch.tensor(
        #     [
        #         odom_msg.pose.pose.orientation.y,
        #         odom_msg.pose.pose.orientation.z * -1, 
        #         odom_msg.pose.pose.orientation.x * -1,
        #         odom_msg.pose.pose.orientation.w, 
        #     ],
        #         device="cuda",
        #         dtype=torch.float16,
        #     ),
        #     iftorch=True,
        # )
        
        # Trans = np.matmul( self.velo2cam_rot,self.imu2velo_rot)
        # # print("########",Trans)
        Trans = np.matmul(self.imu2velo_rot,self.velo2cam_rot)
        Trans_P = np.matmul(Trans,p_num[:3])
        Trans_M = np.matmul(np.linalg.inv(self.imu2velo_rot),M_num[:3, :3])
        
        # print("M0=",M0[:3, :3],"#####Trans###",Trans_M)
        # print("p=",M_num,"#####Trans###",p_num)

        # Trans = np.matmul(Trans, self.imu2velo)
        # Trans = np.matmul(Trans, M_num)

        # Trans = np.matmul(np.linalg.inv(self.camtocam0),np.linalg.inv(self.velo2cam))
        # Trans = np.matmul(Trans,np.linalg.inv(self.imu2velo) )
        # Trans = np.matmul(Trans, Mat)
     
        
     
        M = Trans_M[:3, :3]
        p = Trans_P[:3]

        # p = p_num 

        # p = torch.from_numpy(p).cuda().half()
        # M = torch.from_numpy(M[:3, :3]).cuda().half()
        
        
        other_p = outputs[:, :3]
        # other_M = euler_to_matrix(
        #     outputs[:, -1], outputs[:, :3]
        # )
        
        other_p = (M[:3, :3] @ other_p.T).T + p[:3]
        # other_M = M[:3, :3].unsqueeze(0).float() @ other_M.float()

        # outputs[:, -1] = euler_from_matrix_vec(other_M.half())
        outputs[:, :3] = np.squeeze(other_p, axis=0)
        return outputs


class MatcherType(enum.IntEnum):
    AANET = 0
    MAX = AANET


_MODEL_FACTORY = {MatcherType.AANET: [AANetStereoMatcher, AANetStereoMatcherConfig]}


class StereoMatcherNode(RosNodeBase):
    def __init__(self, internal_rospy):
        RosNodeBase.__init__(self, internal_rospy)
        self._init_parameter()

        if self._matcher_type > MatcherType.MAX:
            self._rospy.logfatal("Not supported stereo matcher")
            sys.exit()

        self._MODEL_CLASS, self._MODEL_CONFIG_CLASS = _MODEL_FACTORY[self._matcher_type]
        self._model_config = self._MODEL_CONFIG_CLASS(rospy)

        if (self._model_config.model_path == "") or (not os.path.isfile(self._model_config.model_path)):
            self._rospy.logfatal("Invalid model path {}".format(self._model_config.model_path))
            sys.exit()

        if self._gpu_idx < 0:
            self._device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                self._rospy.logfatal("GPU environment not available")
                sys.exit()
            else:
                self._device = torch.device("cuda:{}".format(self._gpu_idx))

        self._model = self._MODEL_CLASS(self._model_config, self._device)

        self._disp_pub = self._rospy.Publisher("~disparity", Image, queue_size=1)
        if self._debug:
            self._disp_debug_pub = self._rospy.Publisher("~disp_debug", Image, queue_size=1)

        if self._publish_point_cloud:
            self._pointcloud_pub = self._rospy.Publisher("~pointcloud", PointCloud2, queue_size=1)
            # self.br = tf2_ros.TransformBroadcaster()
            # self.t = TransformStamped()
            # self.t.header.frame_id = "map"
            # self.t.child_frame_id = "camera_gray_left"
            # self.t.transform.rotation.w = 1
        self._subscribe_once()
        self._subscribe()
        self.transforming = ObjectTFConverter()

    def _init_parameter(self):
        self._matcher_type = self._rospy.get_param("~matcher_type", 0)
        self._gpu_idx = self._rospy.get_param("~gpu_idx", -1)
        self._debug = self._rospy.get_param("~debug", True)
        self._img_scale = self._rospy.get_param("~img_scale", 1.0)
        self._disparity_multiplier = self._rospy.get_param("~disparity_multiplier", 256.0)
        self._max_depth = self._rospy.get_param("~max_depth", 30.0)
        self._publish_point_cloud = self._rospy.get_param("~publish_point_cloud", False)
        self._use_raw_img = self._rospy.get_param("~use_raw_img", False)

    def _subscribe(self):

        self._synchronizer_type = self._rospy.get_param("~synchronizer_type", 0)
        self._left_rect_img_sub = message_filters.Subscriber("~left_rect_img", Image)
        self._right_rect_img_sub = message_filters.Subscriber("~right_rect_img", Image)
        self.odom_sub = message_filters.Subscriber("/run_slam/camera_pose", Odometry)

        self._ts = self.to_synchronizer(
            self._synchronizer_type,
            fs=[
                self._left_rect_img_sub,
                self._right_rect_img_sub,
                self.odom_sub,
            ],
            queue_size=10,
            slop=0.1,
        )
        self._ts.registerCallback(self.callback)

    def _subscribe_once(self):
        self._left_camera_info = self._rospy.wait_for_message("~left_camera_info", CameraInfo)
        self._right_camera_info = self._rospy.wait_for_message("~right_camera_info", CameraInfo)
        self._q_matrix = utils.get_q_matrix(self._left_camera_info, self._right_camera_info)
        if not np.isclose(self._img_scale, 1.0):
            scaled_left_camera_info = utils.scale_camera_info(self._left_camera_info, self._img_scale, self._img_scale)
            scaled_right_camera_info = utils.scale_camera_info(
                self._right_camera_info, self._img_scale, self._img_scale
            )
            self._q_matrix = utils.get_q_matrix(scaled_left_camera_info, scaled_right_camera_info)

    def callback(self, left_img_msg: Image, right_img_msg: Image, odom_msg: Odometry):
        # rospy.logwarn(Odometry)
        left_img = utils.to_cv_image(left_img_msg)
        if left_img is None:
            self._rospy.logwarn("Left image empty")
            return

        right_img = utils.to_cv_image(right_img_msg)
        if right_img is None:
            self._rospy.logwarn("Right image empty")
            return

        if self._use_raw_img:
            left_img = utils.remap(left_img, self._left_camera_info)
            right_img = utils.remap(right_img, self._right_camera_info)

        if not np.isclose(self._img_scale, 1.0):
            width = int(left_img.shape[1] * self._img_scale)
            height = int(left_img.shape[0] * self._img_scale)
            resize_dim = (width, height)
            left_img = cv2.resize(left_img, resize_dim, interpolation=cv2.INTER_CUBIC)
            right_img = cv2.resize(right_img, resize_dim, interpolation=cv2.INTER_CUBIC)

        disp_img = self._model.run(left_img, right_img)

        cur_header = self.get_new_header(left_img_msg.header)

        disp_img_msg = utils.to_img_msg(disp_img, "32FC1", cur_header)
        self._disp_pub.publish(disp_img_msg)

        if self._debug:
            disp_img_scaled = disp_img * self._disparity_multiplier
            disp_img_scaled = disp_img_scaled.astype(np.uint16)

            disp_debug_img = cv2.applyColorMap(utils.uint16_to_uint8(disp_img_scaled), cv2.COLORMAP_JET)
            disp_debug_img_msg = utils.to_img_msg(disp_debug_img, "bgr8", cur_header)
            self._disp_debug_pub.publish(disp_debug_img_msg)

        if self._publish_point_cloud:
            projected_points = cv2.reprojectImageTo3D(disp_img, self._q_matrix)
            # rospy.logwarn(type(projected_points))
            projected_points=projected_points.reshape((490*148,3))
            # rospy.logwarn(projected_points.shape)
            # projected_points_torch = torch.from_numpy(projected_points).half().cuda()            
            # rospy.logwarn(projected_points.shape)
            projected_points = self.transforming.transform_pose(projected_points, cur_header, odom_msg)
            # projected_points = projected_points_torch.detach().cpu().numpy()
            projected_points=projected_points.reshape((148,490,3))
            pointcloud_msg = utils.xyzrgb_array_to_pointcloud2(projected_points, left_img, self._max_depth, cur_header)
            self._pointcloud_pub.publish(pointcloud_msg)


def main(argv):
    try:
        rospy.init_node("stereo_matcher", anonymous=False)
        stereo_matcher = StereoMatcherNode(rospy)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("error with stereo_matcher setup")
        sys.exit()


if __name__ == "__main__":
    main(sys.argv)
