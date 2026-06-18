#! /usr/bin/env python3

import rospy
import time
import numpy as np
import copy
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from cdpr_86_msgs.msg import CableLengthsStamped, MotorPositionsStamped
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from jacobian import get_jacobian
from imu_extrinsic import ImuExtrinsic, load_extrinsic_for_node


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def quat_valid(q) -> bool:
    q = np.asarray(q, dtype=float).reshape(4)
    return bool(np.linalg.norm(q) > 1e-9 and np.isfinite(q).all())


class CDPR:

    def __init__(
        self,
        imu_active: bool = False,
        imu_topic: str = "/imu",
        is_calibrated: bool = False,
        use_calibrated_cable_length: bool = None,
        calibration_file: str = None,
        imu_extrinsic_file: str = None,
        apply_imu_extrinsic: bool = True,
        imu_extrinsic: ImuExtrinsic = None,
        publish_cable_lengths: bool = True,
        subscribe_motor_pos: bool = True,
    ):
        self.is_calibrated = bool(is_calibrated)
        if use_calibrated_cable_length is None:
            use_calibrated_cable_length = self.is_calibrated
        self.use_calibrated_cable_length = bool(use_calibrated_cable_length)
        self.calibration_file = calibration_file
        self._kinematic_calibration = None

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
        self._anchorB1 = np.array([0.184, -0.125, 0.110])
        self._anchorB2 = np.array([-0.140, 0.169, -0.110])
        self._anchorB3 = np.array([0.140, 0.169, 0.110])
        self._anchorB4 = np.array([-0.184, -0.125, -0.110])
        self._anchorB5 = np.array([-0.184, 0.125, 0.110])
        self._anchorB6 = np.array([0.140, -0.169, -0.110])
        self._anchorB7 = np.array([-0.140, -0.169, 0.110])
        self._anchorB8 = np.array([0.184, 0.125, -0.110])
        self.b_matrix = np.vstack([self._anchorB1, self._anchorB2, self._anchorB3, self._anchorB4,
                                   self._anchorB5, self._anchorB6, self._anchorB7, self._anchorB8])
        self.cable_radii = np.full(8, 0.025, dtype=float)

        if self.is_calibrated:
            if calibration_file is None:
                calibration_file = Path(__file__).resolve().with_name("cdpr_kinematic_calib.json")
            self._kinematic_calibration = self.load_kinematic_calibration(calibration_file)

        # ros settings
        if rospy.get_name() == '/unnamed':
            rospy.init_node('cdpr_control', anonymous=False)
            print(rospy.get_name())

        try:
            self.use_calibrated_cable_length = _as_bool(
                rospy.get_param("~use_calibrated_cable_length", self.use_calibrated_cable_length)
            )
        except rospy.ROSException:
            pass

        self.publish_cable_lengths = publish_cable_lengths
        self.subscribe_motor_pos = subscribe_motor_pos
        try:
            self.publish_cable_lengths = _as_bool(
                rospy.get_param("~publish_cable_lengths", self.publish_cable_lengths)
            )
            self.subscribe_motor_pos = _as_bool(
                rospy.get_param("~subscribe_motor_pos", self.subscribe_motor_pos)
            )
        except rospy.ROSException:
            pass

        self.imu_active = bool(imu_active)
        self.imu_topic = imu_topic
        self.imu_wait_timeout = float(rospy.get_param("~imu_wait_timeout", 2.0))
        self._imu_quat = None  # scipy quat: [x, y, z, w]
        self._imu_extrinsic = imu_extrinsic
        if self.imu_active and apply_imu_extrinsic and self._imu_extrinsic is None:
            ext_path = imu_extrinsic_file
            if ext_path is None:
                ext_path = rospy.get_param("~imu_extrinsic_file", "cdpr_imu_extrinsic.json")
            self._imu_extrinsic = load_extrinsic_for_node(
                ext_path,
                enabled=True,
                node_name="CDPR",
            )
        if self._imu_extrinsic is not None:
            rospy.loginfo(
                "CDPR IMU extrinsic loaded (n=%d, residual_rms=%.4f deg).",
                self._imu_extrinsic.n_samples,
                self._imu_extrinsic.residual_angle_deg_rms,
            )

        self._velo_pub = rospy.Publisher('motor_velo', Float32MultiArray, queue_size=10)
        if self.publish_cable_lengths:
            self._cable_len_pub = rospy.Publisher(
                'cable_lengths_measure', CableLengthsStamped, queue_size=50
            )
        else:
            self._cable_len_pub = None
            rospy.loginfo("CDPR: cable_lengths_measure publishing disabled.")

        # subscriber and publisher
        self._moving_platform_pose = PoseStamped()  # 末端在基座坐标系的位姿
        rospy.Subscriber('/vrpn_client_node/cdpr/pose', PoseStamped, self._pose_callback, queue_size=1)
        # self._base_frame_pose = PoseStamped()  # 基座坐标系在世界坐标系的位姿
        # rospy.Subscriber('/vrpn_client_node/frame/pose', PoseStamped, self._frame_callback, queue_size=1)

        rospy.Subscriber(self.imu_topic, Imu, self._imu_callback, queue_size=1)

        if self.imu_active:
            try:
                imu_msg = rospy.wait_for_message(self.imu_topic, Imu, timeout=self.imu_wait_timeout)
                self._store_imu_quat(
                    np.array(
                        [
                            imu_msg.orientation.x,
                            imu_msg.orientation.y,
                            imu_msg.orientation.z,
                            imu_msg.orientation.w,
                        ],
                        dtype=float,
                    )
                )
                rospy.loginfo("CDPR got first IMU message from %s before init_cable_length.", self.imu_topic)
            except rospy.ROSException:
                rospy.logwarn(
                    "Timeout waiting for first IMU message on %s (%.2fs), fallback to mocap orientation for init.",
                    self.imu_topic,
                    self.imu_wait_timeout,
                )

        # initial cable lengths and motor positions
        self.motor_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.init_motor_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.init_cable_lens = np.array([0., 0., 0., 0., 0., 0., 0., 0.])  # 初始化的类型必须是浮点！！！

        if self.is_calibrated and self.use_calibrated_cable_length:
            self.init_cable_lens = np.asarray(self._kinematic_calibration["l0"], dtype=float).reshape(8)
            self.init_motor_pos = np.asarray(
                self._kinematic_calibration["init_motor_pos_abs"],
                dtype=float,
            ).reshape(8)
            rospy.loginfo(
                "Using calibrated cable lengths (l0, init_motor_pos_abs) from %s.",
                self.calibration_file,
            )
        else:
            self.init_cable_length()

        self._motor_pos_received = False
        if self.subscribe_motor_pos:
            self.motor_pos_topic = rospy.get_param("~motor_pos_topic", "motor_pos_abs")
            rospy.Subscriber(
                self.motor_pos_topic,
                MotorPositionsStamped,
                self._motor_pos_callback,
                queue_size=1,
            )
            if not (self.is_calibrated and self.use_calibrated_cable_length):
                while not rospy.is_shutdown() and not self._motor_pos_received:
                    rospy.sleep(0.01)
                self.init_motor_pos = self.motor_pos.copy()
                if self.is_calibrated and not self.use_calibrated_cable_length:
                    rospy.loginfo(
                        "is_calibrated=true but use_calibrated_cable_length=false: "
                        "initialized l0 from mocap and init_motor_pos from first motor_pos_abs."
                    )
        else:
            rospy.loginfo("CDPR: motor_pos subscription disabled.")

    def load_kinematic_calibration(self, calibration_file):
        calibration_path = Path(calibration_file).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = Path(__file__).resolve().parent / calibration_path

        with calibration_path.open("r", encoding="utf-8") as f:
            calib = json.load(f)

        a = np.asarray(calib["a"], dtype=float).reshape(8, 3)
        b = np.asarray(calib["b"], dtype=float).reshape(8, 3)
        if "r" in calib:
            self.cable_radii = np.asarray(calib["r"], dtype=float).reshape(8)
        elif "radius" in calib:
            radius = np.asarray(calib["radius"], dtype=float).reshape(-1)
            if radius.size == 1:
                self.cable_radii = np.full(8, float(radius[0]), dtype=float)
            elif radius.size == 8:
                self.cable_radii = radius.astype(float)
            else:
                raise ValueError(f"calibration radius should have 1 or 8 values, got {radius.size}")

        self._anchorA1, self._anchorA2, self._anchorA3, self._anchorA4 = a[0], a[1], a[2], a[3]
        self._anchorA5, self._anchorA6, self._anchorA7, self._anchorA8 = a[4], a[5], a[6], a[7]
        self.a_matrix = a.copy()

        self._anchorB1, self._anchorB2, self._anchorB3, self._anchorB4 = b[0], b[1], b[2], b[3]
        self._anchorB5, self._anchorB6, self._anchorB7, self._anchorB8 = b[4], b[5], b[6], b[7]
        self.b_matrix = b.copy()

        self.calibration_file = str(calibration_path)
        rospy.loginfo("Loaded CDPR kinematic calibration from %s", calibration_path)
        return calib
        
    def init_cable_length(self):
        x0, y0, z0, orient0 = self.wait_for_valid_mocap_pose(timeout=5.0)
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

    def wait_for_valid_mocap_pose(
        self,
        timeout: float = 5.0,
        use_identity_on_timeout: bool = False,
    ):
        """Block until mocap callback has a non-degenerate quaternion, or timeout."""
        rate = rospy.Rate(20.0)
        deadline = rospy.Time.now().to_sec() + float(timeout)
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        while not rospy.is_shutdown():
            x, y, z, quat = self.get_moving_platform_pose_from_mocap()
            q = np.asarray(quat, dtype=float).reshape(4)
            if quat_valid(q):
                rospy.loginfo("CDPR: valid mocap pose received.")
                return float(x), float(y), float(z), q
            if rospy.Time.now().to_sec() > deadline:
                if use_identity_on_timeout:
                    rospy.logwarn(
                        "No valid mocap quaternion within %.1f s; using identity orientation.",
                        timeout,
                    )
                    return float(x), float(y), float(z), identity.copy()
                raise RuntimeError(
                    f"No valid mocap quaternion within {timeout:.1f} s "
                    "(VRPN may be down or not streaming)."
                )
            rate.sleep()
        x, y, z, quat = self.get_moving_platform_pose_from_mocap()
        q = np.asarray(quat, dtype=float).reshape(4)
        if quat_valid(q):
            return float(x), float(y), float(z), q
        if use_identity_on_timeout:
            return float(x), float(y), float(z), identity.copy()
        raise RuntimeError("Shutdown before valid mocap pose.")

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

    def _motor_pos_callback(self, data: MotorPositionsStamped):
        self.motor_pos = np.array(data.positions, dtype=float)
        self._motor_pos_received = True

        if not self.publish_cable_lengths or self._cable_len_pub is None:
            return

        cable_msg = CableLengthsStamped()
        cable_msg.header.stamp = (
            data.header.stamp if data.header.stamp != rospy.Time() else rospy.Time.now()
        )
        cable_msg.header.frame_id = data.header.frame_id or "world"
        cable_msg.lengths = self.calculate_cable_length_from_motor_pos(self.motor_pos).tolist()
        self._cable_len_pub.publish(cable_msg)

    def _correct_imu_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        if self._imu_extrinsic is None:
            return q
        corrected = self._imu_extrinsic.apply_quat(q)
        return corrected if corrected is not None else q

    def _store_imu_quat(self, quat_xyzw: np.ndarray) -> None:
        self._imu_quat = self._correct_imu_quat(quat_xyzw)

    def _imu_callback(self, data: Imu):
        self._store_imu_quat(
            np.array(
                [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w],
                dtype=float,
            )
        )

    def calculate_cable_length_from_motor_pos(self, motor_pos):
        cable_lengths = self.init_cable_lens.copy()
        for i in range(min(len(cable_lengths), len(motor_pos))):
            motor_delta = (motor_pos[i] - self.init_motor_pos[i]) / 10000.0 * 2.0 * np.pi
            if not i % 2:
                cable_lengths[i] = self.init_cable_lens[i] - motor_delta * self.cable_radii[i]
            else:
                cable_lengths[i] = self.init_cable_lens[i] + motor_delta * self.cable_radii[i]
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



    for i in range(1):
        cdpr.set_motor_velo([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
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

