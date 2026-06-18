#!/usr/bin/env python3

import threading
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rospy
from cdpr_86_msgs.msg import CableLengthsStamped
from cdpr_euler_ekf import cdpr_geometry_from_calibration_file, make_demo_geometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R

from imu_extrinsic import ImuExtrinsic, load_extrinsic_for_node


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class RollingBuffer:
    def __init__(self, window_sec: float, dim: int = 6):
        self.window_sec = window_sec
        self.dim = dim
        self.t = deque()
        self.values = [deque() for _ in range(dim)]

    def append(self, stamp_sec: float, vec6: np.ndarray) -> None:
        self.t.append(stamp_sec)
        for i in range(self.dim):
            self.values[i].append(float(vec6[i]))
        self.trim(stamp_sec)

    def trim(self, newest_sec: float) -> None:
        threshold = newest_sec - self.window_sec
        while self.t and self.t[0] < threshold:
            self.t.popleft()
            for i in range(self.dim):
                self.values[i].popleft()


class PoseImuNavPlotNode:
    def __init__(self):
        self.window_sec = float(rospy.get_param("~window_sec", 10.0))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 10.0))
        self.pose_topic = rospy.get_param("~pose_topic", "/vrpn_client_node/cdpr/pose")
        self.ekf_topic = rospy.get_param("~ekf_topic", "/ekf_pose")
        self.fk_topic = rospy.get_param("~fk_topic", "/fk_pose")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.cable_topic = rospy.get_param("~cable_topic", "/cable_lengths_measure")
        self.show_cable_plot = _as_bool(rospy.get_param("~show_cable_plot", False))
        self.rpy_from_imu = _as_bool(rospy.get_param("~rpy_from_imu", False))
        self.apply_imu_extrinsic = _as_bool(rospy.get_param("~apply_imu_extrinsic", True))
        self.imu_extrinsic_file = rospy.get_param("~imu_extrinsic_file", "cdpr_imu_extrinsic.json")
        self._imu_extrinsic: ImuExtrinsic = None
        if self.rpy_from_imu and self.apply_imu_extrinsic:
            self._imu_extrinsic = load_extrinsic_for_node(
                self.imu_extrinsic_file,
                enabled=True,
                node_name="compare_plot",
            )
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        # True: IMU linear_acceleration is specific force f=a-g (default, common raw IMU output).
        # False: IMU linear_acceleration is already linear acceleration a (gravity removed).
        self.imu_acc_is_specific_force = _as_bool(rospy.get_param("~imu_acc_is_specific_force", True))
        # Match cdpr_euler_ekf_ros_node: same ~is_calibrated / ~calibration_file (no CDPR() here → no extra pubs).
        self.is_calibrated = _as_bool(rospy.get_param("~is_calibrated", True))
        self.calibration_file = rospy.get_param("~calibration_file", "cdpr_kinematic_calib.json")
        if self.is_calibrated:
            self.geom = cdpr_geometry_from_calibration_file(
                self.calibration_file,
                base_dir=Path(__file__).resolve().parent,
            )
        else:
            self.geom = make_demo_geometry(use_ros_cdpr=False)
        rospy.loginfo(
            "compare_plot geometry: is_calibrated=%s file=%s",
            str(self.is_calibrated),
            str(self.calibration_file) if self.is_calibrated else "(nominal)",
        )

        self.lock = threading.Lock()
        self.pose_buf = RollingBuffer(self.window_sec)
        self.ekf_buf = RollingBuffer(self.window_sec)
        self.nav_buf = RollingBuffer(self.window_sec)
        self.fk_buf = RollingBuffer(self.window_sec)
        self.cable_buf = RollingBuffer(self.window_sec, dim=self.geom.m)
        # Plot visibility switch: change True/False here to show/hide each data source.
        self.show_series = {
            "mocap_pose": True,
            "ekf_pose": True,
            "imu_nav": False,
            "lm_fk": True,
        }

        # Inertial navigation state (simple strapdown integration).
        self.nav_p = np.zeros(3, dtype=float)
        self.nav_v = np.zeros(3, dtype=float)
        self.nav_rpy = np.zeros(3, dtype=float)  # roll pitch yaw
        self.last_imu_t = None
        self.nav_initialized = False
        self.latest_imu_quat = None  # [x, y, z, w], used for optional rpy override

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=100)
        rospy.Subscriber(self.ekf_topic, PoseStamped, self.ekf_callback, queue_size=100)
        rospy.Subscriber(self.fk_topic, PoseStamped, self.fk_callback, queue_size=100)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=300)
        rospy.Subscriber(self.cable_topic, CableLengthsStamped, self.cable_callback, queue_size=100)

        self.fig, self.axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        self.fig.suptitle("Pose/EKF/IMU-INS (Rolling 10s Window)")
        self.labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
        self.lines_pose = []
        self.lines_ekf = []
        self.lines_nav = []
        self.lines_fk = []
        for i, ax in enumerate(self.axes):
            line_pose, = ax.plot([], [], "b-", linewidth=1.5, label="mocap_pose")
            line_ekf, = ax.plot([], [], "r-", linewidth=1.5, label="ekf_pose")
            line_nav, = ax.plot([], [], "g-", linewidth=1.5, label="imu_nav")
            line_fk, = ax.plot([], [], "m-", linewidth=1.2, label="lm_fk")
            self.lines_pose.append(line_pose)
            self.lines_ekf.append(line_ekf)
            self.lines_nav.append(line_nav)
            self.lines_fk.append(line_fk)
            line_pose.set_visible(self.show_series["mocap_pose"])
            line_ekf.set_visible(self.show_series["ekf_pose"])
            line_nav.set_visible(self.show_series["imu_nav"])
            line_fk.set_visible(self.show_series["lm_fk"])
            ax.set_ylabel(self.labels[i])
            ax.grid(True, alpha=0.3)
        self.axes[-1].set_xlabel("time [s], relative to now")
        self.axes[0].legend(loc="upper left")
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.97])

        self.fig_cable = None
        self.axes_cable = None
        self.lines_cable = []
        if self.show_cable_plot:
            self.fig_cable, self.axes_cable = plt.subplots(self.geom.m, 1, figsize=(12, 12), sharex=True)
            self.fig_cable.suptitle("Cable Lengths Measure (Rolling 10s Window)")
            for i, ax in enumerate(self.axes_cable):
                line_cable, = ax.plot([], [], linewidth=1.5, label=f"cable_{i + 1}")
                self.lines_cable.append(line_cable)
                ax.set_ylabel(f"L{i + 1} [m]")
                ax.grid(True, alpha=0.3)
            self.axes_cable[-1].set_xlabel("time [s], relative to now")
            self.axes_cable[0].legend(loc="upper left")
            self.fig_cable.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.ion()
        plt.show()

        rospy.loginfo("Pose/IMU nav plot node started.")
        rospy.loginfo(
            "topics: pose=%s, ekf=%s, imu=%s, cable=%s",
            self.pose_topic,
            self.ekf_topic,
            self.imu_topic,
            self.cable_topic,
        )
        rospy.loginfo("fk_topic=%s", self.fk_topic)
        rospy.loginfo("imu_acc_is_specific_force=%s", str(self.imu_acc_is_specific_force))
        rospy.loginfo("show_cable_plot=%s", str(self.show_cable_plot))
        rospy.loginfo("rpy_from_imu=%s", str(self.rpy_from_imu))
        rospy.loginfo(
            "IMU extrinsic: apply=%s file=%s loaded=%s",
            str(self.apply_imu_extrinsic),
            self.imu_extrinsic_file,
            str(self._imu_extrinsic is not None),
        )

    def _correct_imu_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        if self._imu_extrinsic is None:
            return q
        out = self._imu_extrinsic.apply_quat(q)
        return out if out is not None else q

    def _correct_imu_vector(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=float).reshape(3)
        if self._imu_extrinsic is None:
            return v
        return self._imu_extrinsic.apply_vector(v)

    @staticmethod
    def pose_to_xyzrpy(msg: PoseStamped) -> np.ndarray:
        p = msg.pose.position
        q = msg.pose.orientation
        yaw, pitch, roll = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("ZYX", degrees=False)
        return np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=float)

    def pose_callback(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        vec = self.pose_to_xyzrpy(msg)
        # Optional: keep mocap xyz, but override plotted rpy with IMU attitude.
        if self.rpy_from_imu and self.latest_imu_quat is not None:
            yaw, pitch, roll = R.from_quat(self.latest_imu_quat).as_euler("ZYX", degrees=False)
            vec[3:] = np.array([roll, pitch, yaw], dtype=float)
        with self.lock:
            self.pose_buf.append(t, vec)

    def ekf_callback(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        vec = self.pose_to_xyzrpy(msg)
        with self.lock:
            self.ekf_buf.append(t, vec)

    def fk_callback(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        vec = self.pose_to_xyzrpy(msg)
        with self.lock:
            self.fk_buf.append(t, vec)

    def imu_callback(self, msg: Imu) -> None:
        t = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        self.latest_imu_quat = self._correct_imu_quat(
            np.array(
                [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                dtype=float,
            )
        )
        if self.last_imu_t is None:
            self.last_imu_t = t
            self.nav_initialized = True
            return

        dt = t - self.last_imu_t
        if dt <= 1e-6 or dt > 0.2:
            self.last_imu_t = t
            return
        self.last_imu_t = t

        gyro_b = self._correct_imu_vector(
            np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        )
        imu_acc_b = self._correct_imu_vector(
            np.array(
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                dtype=float,
            )
        )

        # Integrate attitude with body rates using Euler angles (small-dt approximation).
        self.nav_rpy += gyro_b * dt
        self.nav_rpy = (self.nav_rpy + np.pi) % (2.0 * np.pi) - np.pi

        # Convert IMU acceleration to world linear acceleration and integrate position/velocity.
        # If IMU gives specific force f=a-g, recover linear acceleration with: a = C_wb f + g.
        # If IMU already gives linear acceleration, just rotate to world frame.
        c_wb = R.from_euler("ZYX", [self.nav_rpy[2], self.nav_rpy[1], self.nav_rpy[0]], degrees=False).as_matrix()
        if self.imu_acc_is_specific_force:
            acc_w = c_wb @ imu_acc_b + self.gravity
        else:
            acc_w = c_wb @ imu_acc_b
        self.nav_v += acc_w * dt
        self.nav_p += self.nav_v * dt

        nav_vec = np.hstack([self.nav_p, self.nav_rpy])
        with self.lock:
            self.nav_buf.append(t, nav_vec)

    def cable_callback(self, msg: CableLengthsStamped) -> None:
        t = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        lengths = np.asarray(msg.lengths, dtype=float)
        if lengths.size != self.geom.m:
            rospy.logwarn_throttle(
                2.0,
                "LM FK got cable size %d, expected %d, skip.",
                lengths.size,
                self.geom.m,
            )
            return
        with self.lock:
            self.cable_buf.append(t, lengths)

    def update_plot(self) -> None:
        now = rospy.Time.now().to_sec()
        with self.lock:
            self.pose_buf.trim(now)
            self.ekf_buf.trim(now)
            self.nav_buf.trim(now)
            self.fk_buf.trim(now)
            self.cable_buf.trim(now)

            t_pose = np.array(self.pose_buf.t, dtype=float)
            t_ekf = np.array(self.ekf_buf.t, dtype=float)
            t_nav = np.array(self.nav_buf.t, dtype=float)
            t_fk = np.array(self.fk_buf.t, dtype=float)
            t_cable = np.array(self.cable_buf.t, dtype=float)

            x_pose = t_pose - now if t_pose.size > 0 else np.array([])
            x_ekf = t_ekf - now if t_ekf.size > 0 else np.array([])
            x_nav = t_nav - now if t_nav.size > 0 else np.array([])
            x_fk = t_fk - now if t_fk.size > 0 else np.array([])
            x_cable = t_cable - now if t_cable.size > 0 else np.array([])

            for i in range(6):
                y_pose = np.array(self.pose_buf.values[i], dtype=float)
                y_ekf = np.array(self.ekf_buf.values[i], dtype=float)
                y_nav = np.array(self.nav_buf.values[i], dtype=float)
                y_fk = np.array(self.fk_buf.values[i], dtype=float)
                if i >= 3:  # roll/pitch/yaw in degree for readability
                    y_pose = np.rad2deg(y_pose)
                    y_ekf = np.rad2deg(y_ekf)
                    y_nav = np.rad2deg(y_nav)
                    y_fk = np.rad2deg(y_fk)

                self.lines_pose[i].set_data(x_pose, y_pose)
                self.lines_ekf[i].set_data(x_ekf, y_ekf)
                self.lines_nav[i].set_data(x_nav, y_nav)
                self.lines_fk[i].set_data(x_fk, y_fk)

                ax = self.axes[i]
                ax.set_xlim(-self.window_sec, 0.0)
                y_active = []
                if self.show_series["mocap_pose"]:
                    y_active.append(y_pose)
                if self.show_series["ekf_pose"]:
                    y_active.append(y_ekf)
                if self.show_series["imu_nav"]:
                    y_active.append(y_nav)
                if self.show_series["lm_fk"]:
                    y_active.append(y_fk)
                y_all = (
                    np.hstack(y_active)
                    if len(y_active) > 0 and sum(arr.size for arr in y_active) > 0
                    else np.array([0.0])
                )
                y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
                span = y_max - y_min
                if i < 3:
                    # Keep xyz axis at least 0.02 m range; expand with data when larger.
                    min_span = 0.02
                    center = 0.5 * (y_min + y_max)
                    span = max(span, min_span)
                    half = 0.5 * span
                    ax.set_ylim(center - half, center + half)
                    # if span < 1e-6:
                    #     y_min -= 1.0
                    #     y_max += 1.0
                    #     span = y_max - y_min
                    # pad = 0.1 * span
                    # ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    # Keep xyz axis at least 0.02 m range; expand with data when larger.
                    min_span = 0.2
                    center = 0.5 * (y_min + y_max)
                    span = max(span, min_span)
                    half = 0.5 * span
                    ax.set_ylim(center - half, center + half)
                    # if span < 1e-6:
                    #     y_min -= 1.0
                    #     y_max += 1.0
                    #     span = y_max - y_min
                    # pad = 0.1 * span
                    # ax.set_ylim(y_min - pad, y_max + pad)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self.show_cable_plot:
            for i in range(self.geom.m):
                y_cable = np.array(self.cable_buf.values[i], dtype=float)
                self.lines_cable[i].set_data(x_cable, y_cable)
                ax_cable = self.axes_cable[i]
                ax_cable.set_xlim(-self.window_sec, 0.0)
                if y_cable.size > 0:
                    y_min, y_max = float(np.min(y_cable)), float(np.max(y_cable))
                else:
                    y_min, y_max = 0.0, 1.0
                if abs(y_max - y_min) < 1e-6:
                    y_min -= 1.0
                    y_max += 1.0
                pad = 0.1 * (y_max - y_min)
                ax_cable.set_ylim(y_min - pad, y_max + pad)
            self.fig_cable.canvas.draw_idle()
            self.fig_cable.canvas.flush_events()

    def run(self) -> None:
        rate = rospy.Rate(self.refresh_hz)
        while not rospy.is_shutdown():
            self.update_plot()
            plt.pause(0.001)
            rate.sleep()


def main():
    rospy.init_node("pose_imu_nav_plot", anonymous=False)
    node = PoseImuNavPlotNode()
    node.run()


if __name__ == "__main__":
    main()
