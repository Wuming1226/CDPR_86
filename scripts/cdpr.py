#! /usr/bin/env python3

import rospy
import time
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

from cdpr_86_host.msg import CableLengthsStamped
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from jacobian import get_jacobian


class CDPR:

    def __init__(self, imu_active: bool = False, imu_topic: str = "/imu"):

        # 坐标系marker点在基座坐标下的位置偏移
        self.x_offset = -1.020
        self.y_offset = -1.020
        self.z_offset = 1.552
        self.pos_off = np.array([self.x_offset, self.y_offset, self.z_offset])

        # anchor positions in the base frame
        self._anchorA1 = np.array([-0.260, -0.243, 2.300])
        self._anchorA2 = np.array([-0.361, -0.125, 2.300])
        self._anchorA3 = np.array([-2.049, -0.089, 2.300])
        self._anchorA4 = np.array([-2.169, -0.212, 2.300])
        self._anchorA5 = np.array([-2.193, -1.225, 2.290])
        self._anchorA6 = np.array([-2.084, -1.357, 2.300])
        self._anchorA7 = np.array([-0.415, -1.384, 2.300])
        self._anchorA8 = np.array([-0.290, -1.252, 2.300])
        self.a_matrix = np.vstack([self._anchorA1, self._anchorA2, self._anchorA3, self._anchorA4,
                                   self._anchorA5, self._anchorA6, self._anchorA7, self._anchorA8])

        # self._anchorB1 = np.array([0.0117, -0.0075, 0.0065])
        # self._anchorB2 = np.array([-0.0085, 0.0107, -0.0065])
        # self._anchorB3 = np.array([0.0085, 0.0107, 0.0065])
        # self._anchorB4 = np.array([-0.0117, -0.0075, -0.0065])
        # self._anchorB5 = np.array([-0.0117, 0.0075, 0.0065])
        # self._anchorB6 = np.array([0.0085, -0.0107, -0.0068])
        # self._anchorB7 = np.array([-0.0085, -0.0107, 0.0065])
        # self._anchorB8 = np.array([0.0117, 0.0075, -0.0065])
        self._anchorB1 = np.array([0.0184, -0.0125, 0.0110])
        self._anchorB2 = np.array([-0.0140, 0.0169, -0.0110])
        self._anchorB3 = np.array([0.0140, 0.0169, 0.0110])
        self._anchorB4 = np.array([-0.0184, -0.0125, -0.0110])
        self._anchorB5 = np.array([-0.0184, 0.0125, 0.0110])
        self._anchorB6 = np.array([0.0140, -0.0169, -0.0110])
        self._anchorB7 = np.array([-0.0140, -0.0169, 0.0110])
        self._anchorB8 = np.array([0.0184, 0.0125, -0.0110])
        self.b_matrix = np.vstack([self._anchorB1, self._anchorB2, self._anchorB3, self._anchorB4,
                                   self._anchorB5, self._anchorB6, self._anchorB7, self._anchorB8])

        # ros settings
        if rospy.get_name() == '/unnamed':
            rospy.init_node('cdpr_control', anonymous=False)
            print(rospy.get_name())

        self.imu_active = bool(imu_active)
        self.imu_topic = imu_topic
        self.imu_wait_timeout = float(rospy.get_param("~imu_wait_timeout", 2.0))
        self._imu_quat = None  # scipy quat: [x, y, z, w]

        self._velo_pub = rospy.Publisher('motor_velo', Float32MultiArray, queue_size=10)
        self._cable_len_pub = rospy.Publisher('cable_lengths_measure', CableLengthsStamped, queue_size=50)

        # subscriber and publisher
        self._moving_platform_pose = PoseStamped()  # 末端在基座坐标系的位姿
        rospy.Subscriber('/vrpn_client_node/cdpr/pose', PoseStamped, self._pose_callback, queue_size=1)
        # self._base_frame_pose = PoseStamped()  # 基座坐标系在世界坐标系的位姿
        # rospy.Subscriber('/vrpn_client_node/frame/pose', PoseStamped, self._frame_callback, queue_size=1)

        rospy.Subscriber(self.imu_topic, Imu, self._imu_callback, queue_size=1)

        if self.imu_active:
            try:
                imu_msg = rospy.wait_for_message(self.imu_topic, Imu, timeout=self.imu_wait_timeout)
                self._imu_quat = np.array(
                    [
                        imu_msg.orientation.x,
                        imu_msg.orientation.y,
                        imu_msg.orientation.z,
                        imu_msg.orientation.w,
                    ],
                    dtype=float,
                )
                rospy.loginfo("CDPR got first IMU message from %s before init_cable_length.", self.imu_topic)
            except rospy.ROSException:
                rospy.logwarn(
                    "Timeout waiting for first IMU message on %s (%.2fs), fallback to mocap orientation for init.",
                    self.imu_topic,
                    self.imu_wait_timeout,
                )

        # initial cable lengths and motor positions
        self.init_cable_lens = np.array([0., 0., 0., 0., 0., 0., 0., 0.])  # 初始化的类型必须是浮点！！！
        self.init_cable_length()
        self.motor_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.init_motor_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self._motor_pos_received = False
        rospy.Subscriber('motor_pos_rel', Float32MultiArray, self._motor_pos_callback, queue_size=1)
        while not rospy.is_shutdown() and not self._motor_pos_received:
            rospy.sleep(0.01)
        self.init_motor_pos = self.motor_pos.copy()
        
    def init_cable_length(self):
        # calculate origin cable lengths
        time.sleep(1)
        x0, y0, z0, orient0 = self.get_moving_platform_pose_from_mocap()
        pos0 = np.array([x0, y0, z0])

        rot0 = R.from_quat(orient0)
        b_matrix = rot0.apply(self.b_matrix)

        self.init_cable_lens[0] = np.linalg.norm(pos0 - self._anchorA1 + b_matrix[0, :])
        self.init_cable_lens[1] = np.linalg.norm(pos0 - self._anchorA2 + b_matrix[1, :])
        self.init_cable_lens[2] = np.linalg.norm(pos0 - self._anchorA3 + b_matrix[2, :])
        self.init_cable_lens[3] = np.linalg.norm(pos0 - self._anchorA4 + b_matrix[3, :])
        self.init_cable_lens[4] = np.linalg.norm(pos0 - self._anchorA5 + b_matrix[4, :])
        self.init_cable_lens[5] = np.linalg.norm(pos0 - self._anchorA6 + b_matrix[5, :])
        self.init_cable_lens[6] = np.linalg.norm(pos0 - self._anchorA7 + b_matrix[6, :])
        self.init_cable_lens[7] = np.linalg.norm(pos0 - self._anchorA8 + b_matrix[7, :])
        print("init_cable_lens: {}".format(self.init_cable_lens))

    def _pose_callback(self, data):
        # if motion data is lost(999999), do not update
        if (np.abs(data.pose.position.x) > 2000 or np.abs(data.pose.position.y) > 2000
                or np.abs(data.pose.position.z) > 2000):
            pass
        # elif self._base_frame_pose.pose.orientation.w == 0:  # 等待 frame_pose 先初始化
        #     pass
        else:
            # pose
            # self._moving_platform_pose.pose.position.x = data.pose.position.x - self._base_frame_pose.pose.position.x
            # self._moving_platform_pose.pose.position.y = data.pose.position.y - self._base_frame_pose.pose.position.y
            # self._moving_platform_pose.pose.position.z = data.pose.position.z - self._base_frame_pose.pose.position.z
            # quat = (R.from_quat(
            #     [self._base_frame_pose.pose.orientation.x, self._base_frame_pose.pose.orientation.y,
            #      self._base_frame_pose.pose.orientation.z,
            #      self._base_frame_pose.pose.orientation.w]).inv() * R.from_quat(
            #     [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
            #      data.pose.orientation.w])).as_quat()
            # self._moving_platform_pose.pose.orientation.x = quat[0]
            # self._moving_platform_pose.pose.orientation.y = quat[1]
            # self._moving_platform_pose.pose.orientation.z = quat[2]
            # self._moving_platform_pose.pose.orientation.w = quat[3]
            self._moving_platform_pose.pose.position.x = data.pose.position.x
            self._moving_platform_pose.pose.position.y = data.pose.position.y
            self._moving_platform_pose.pose.position.z = data.pose.position.z
            self._moving_platform_pose.pose.orientation = data.pose.orientation
            # header
            self._moving_platform_pose.header.frame_id = data.header.frame_id
            self._moving_platform_pose.header.stamp = data.header.stamp

    # # 基座坐标姿态回调函数
    # def _frame_callback(self, data):
    #     # if motion data is lost(999999), do not update
    #     if (np.abs(data.pose.position.x) > 2000 or np.abs(data.pose.position.y) > 2000
    #             or np.abs(data.pose.position.z) > 2000):
    #         pass
    #     else:
    #         # pose
    #         self._base_frame_pose.pose.position.x = data.pose.position.x - self.x_offset
    #         self._base_frame_pose.pose.position.y = data.pose.position.y - self.y_offset
    #         self._base_frame_pose.pose.position.z = data.pose.position.z - self.z_offset
    #         self._base_frame_pose.pose.orientation = data.pose.orientation
    #
    #         # header
    #         self._base_frame_pose.header.frame_id = data.header.frame_id
    #         self._base_frame_pose.header.stamp = data.header.stamp

    def _motor_pos_callback(self, data):
        self.motor_pos = np.array(data.data, dtype=float)
        self._motor_pos_received = True

        cable_msg = CableLengthsStamped()
        cable_msg.header.stamp = rospy.Time.now()
        cable_msg.header.frame_id = "world"
        cable_msg.lengths = self.calculate_cable_length_from_motor_pos(self.motor_pos).tolist()
        self._cable_len_pub.publish(cable_msg)

    def _imu_callback(self, data: Imu):
        self._imu_quat = np.array(
            [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w],
            dtype=float,
        )

    def calculate_cable_length_from_motor_pos(self, motor_pos):
        r = 0.025
        cable_lengths = self.init_cable_lens.copy()
        for i in range(min(len(cable_lengths), len(motor_pos))):
            if not i % 2:
                cable_lengths[i] = self.init_cable_lens[i] - motor_pos[i] * r
            else:
                cable_lengths[i] = self.init_cable_lens[i] + motor_pos[i] * r
        return cable_lengths

    def set_motor_velo(self, motor_velo):
        motor_velo = Float32MultiArray(data=np.array(motor_velo))
        self._velo_pub.publish(motor_velo)

    # def set_cable_lens(self, cable_lens):
    #     r = 0.025
    #     pos_com = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    #     for i in range(8):
    #         if not i % 2:
    #             pos_com[i] = -(cable_lens[i] - self.init_cable_lens[i]) / r
    #         else:
    #             pos_com[i] = (cable_lens[i] - self.init_cable_lens[i]) / r
    #     motor_pos_com = Float32MultiArray(data=pos_com)

    #     self._pos_com_pub.publish(motor_pos_com)

    def get_moving_platform_pose_from_mocap(self):
        x = self._moving_platform_pose.pose.position.x
        y = self._moving_platform_pose.pose.position.y
        z = self._moving_platform_pose.pose.position.z

        mocap_quat = [
            self._moving_platform_pose.pose.orientation.x,
            self._moving_platform_pose.pose.orientation.y,
            self._moving_platform_pose.pose.orientation.z,
            self._moving_platform_pose.pose.orientation.w,
        ]

        quat = mocap_quat
        if self.imu_active and self._imu_quat is not None:
            quat = self._imu_quat.tolist()

        return (x, y, z, quat)

    def calculate_cable_length_at_pose(self, pos, rot):
        pos0 = np.array(pos)
        b_matrix = rot.apply(self.b_matrix)

        return np.array([np.linalg.norm(pos0 - self._anchorA1 + b_matrix[0, :]),
                         np.linalg.norm(pos0 - self._anchorA2 + b_matrix[1, :]),
                         np.linalg.norm(pos0 - self._anchorA3 + b_matrix[2, :]),
                         np.linalg.norm(pos0 - self._anchorA4 + b_matrix[3, :]),
                         np.linalg.norm(pos0 - self._anchorA5 + b_matrix[4, :]),
                         np.linalg.norm(pos0 - self._anchorA6 + b_matrix[5, :]),
                         np.linalg.norm(pos0 - self._anchorA7 + b_matrix[6, :]),
                         np.linalg.norm(pos0 - self._anchorA8 + b_matrix[7, :])
                         ])

    def get_cable_attachment_points(self):
        return self.a_matrix.copy(), self.b_matrix.copy()


    # def get_cable_length(self):
    #     r = 0.025
    #     cable_length = np.array([0.0, 0.0, 0.0, 0.0])
    #     cable_length[0] = self.motor_pos[0] * r + self.init_cable_lens[0]
    #     cable_length[1] = self.motor_pos[1] * r + self.init_cable_lens[1]
    #     cable_length[2] = self.motor_pos[2] * r + self.init_cable_lens[2]
    #     cable_length[3] = self.motor_pos[3] * r + self.init_cable_lens[3]
    #     return cable_length

    # def pre_tension(self, cable1_flag, cable2_flag, cable3_flag, cable4_flag):
    #
    #     if cable1_flag:
    #         time.sleep(0.5)
    #         # cable1 pre-tightening
    #         print('cable1 pre-tension...')
    #         self.set_motor_velo(-50, 0, 0, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose_from_mocap()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose_from_mocap()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable2_flag:
    #         time.sleep(0.5)
    #         # cable2 pre-tightening
    #         print('cable2 pre-tension...')
    #         self.set_motor_velo(0, -50, 0, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose_from_mocap()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose_from_mocap()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable3_flag:
    #         time.sleep(0.5)
    #         # cable3 pre-tightening
    #         print('cable3 pre-tension...')
    #         self.set_motor_velo(0, 0, -50, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose_from_mocap()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose_from_mocap()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable4_flag:
    #         time.sleep(0.5)
    #         # cable4 pre-tightening
    #         print('cable4 pre-tension...')
    #         self.set_motor_velo(0, 0, 0, -50)
    #         x0, y0, z0, _ = self.get_moving_platform_pose_from_mocap()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose_from_mocap()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)

    # def loosen(self):
    #     print('loosening...')
    #     self.set_motor_velo(0.3, 0.3, 0.3, 0.3)
    #     time.sleep(0.2)
    #     self.set_motor_velo(0, 0, 0, 0)
    #     time.sleep(0.5)


if __name__ == "__main__":

    cdpr = CDPR()
    # rate = rospy.Rate(20)
    # cdpr.pretighten(True, True, True, True)
    # cdpr.init_cable_length(True, True, True, True)
    time.sleep(1)
    print('start')
    # cdpr.set_motor_velo([0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)
    # cdpr.set_motor_velo([0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    # time.sleep(1)

    # cdpr.set_motor_velo([0., 0., 0., -0.2, 0., 0., 0., -0.2])
    # # print("target speed: [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3]")
    # time.sleep(0.2)

    # for i in range(16):
    #     cdpr.set_motor_velo([0., 0.3, -0., 0., -0., 0., 0., -0.])
    #     time.sleep(0.4)



    # cdpr.set_motor_velo([-0., 0., -0.5, 0.5, -0., 0., -0., 0.])
    # time.sleep(0.2)



    for i in range(4):
        cdpr.set_motor_velo([0., 0.5, 0, 0, 0, 0, 0, 0])
        time.sleep(0.2)
    cdpr.set_motor_velo([0, 0, 0, 0, 0, 0, 0, 0])
    print('end')
    # cdpr.set_motor_velo(0.1, 0.1, 0.1, 0.1)
    # start_time = time.time()
    # while time.time() - start_time < 2:
    #     cdpr.set_motor_velo(0.1, 0.1, 0.1, 0.1)
    #     print(cdpr._moving_platform_pose.pose)
    #     time.sleep(0.05)
    # cdpr.set_motor_velo(-0.1, -0.1, -0.1, -0.1)
    # start_time = time.time()
    # while time.time() - start_time < 2:
    #     x, y, z, quat = cdpr.get_moving_platform_pose_from_mocap()
    #     rot = R.from_quat(quat)  # 当前姿态
    #     euler = rot.as_euler('ZYX', degrees=False)
    #     print(euler * 180 / np.pi)
    #     time.sleep(0.05)
    # cdpr.set_motor_velo(0, 0, 0, 0)

