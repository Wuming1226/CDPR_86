#!/usr/bin/env python3
"""
Rolling-window plot: raw IMU, calibrated IMU, and mocap roll / pitch / yaw.

Calibrated IMU uses cdpr_imu_extrinsic.json (same as EKF nodes).

Usage:
  rosrun cdpr_86_host plot_imu_mocap_rpy.py
  rosrun cdpr_86_host plot_imu_mocap_rpy.py _imu_extrinsic_file:=cdpr_imu_extrinsic.json
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu

from imu_extrinsic import ImuExtrinsic, default_extrinsic_path, load_imu_extrinsic, resolve_extrinsic_path


def quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        return np.full(3, np.nan)
    q = q / np.linalg.norm(q)
    yaw, pitch, roll = R.from_quat(q).as_euler("ZYX", degrees=False)
    return np.array([roll, pitch, yaw], dtype=float)


def wrap_angle_arr(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


class RollingRpyBuffer:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self.t: Deque[float] = deque()
        self.roll: Deque[float] = deque()
        self.pitch: Deque[float] = deque()
        self.yaw: Deque[float] = deque()

    def append(self, stamp_sec: float, rpy: np.ndarray) -> None:
        self.t.append(stamp_sec)
        self.roll.append(float(rpy[0]))
        self.pitch.append(float(rpy[1]))
        self.yaw.append(float(rpy[2]))
        self.trim(stamp_sec)

    def trim(self, newest_sec: float) -> None:
        threshold = newest_sec - self.window_sec
        while self.t and self.t[0] < threshold:
            self.t.popleft()
            self.roll.popleft()
            self.pitch.popleft()
            self.yaw.popleft()

    def arrays(self, unwrap: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.t:
            return np.array([]), np.array([]), np.array([]), np.array([])
        t_arr = np.asarray(self.t, dtype=float)
        roll = np.asarray(self.roll, dtype=float)
        pitch = np.asarray(self.pitch, dtype=float)
        yaw = np.asarray(self.yaw, dtype=float)
        if unwrap:
            roll = np.unwrap(roll)
            pitch = np.unwrap(pitch)
            yaw = np.unwrap(yaw)
        return t_arr, roll, pitch, yaw


def _stamp_sec(header) -> float:
    if header.stamp != rospy.Time():
        return header.stamp.to_sec()
    return rospy.Time.now().to_sec()


def nearest_indices(t_query: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    if t_query.size == 0 or t_ref.size == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(t_ref, t_query)
    idx = np.clip(idx, 0, len(t_ref) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_ref) - 1)
    left = np.abs(t_ref[idx0] - t_query)
    right = np.abs(t_ref[idx] - t_query)
    return np.where(left <= right, idx0, idx)


class ImuMocapRpyPlotNode:
    def __init__(self) -> None:
        self.window_sec = float(rospy.get_param("~window_sec", 10.0))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 20.0))
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.mocap_topic = rospy.get_param("~mocap_topic", "/vrpn_client_node/cdpr/pose")
        self.plot_in_degrees = bool(rospy.get_param("~plot_in_degrees", True))
        self.unwrap_angles = bool(rospy.get_param("~unwrap_angles", True))
        self.show_delta = bool(rospy.get_param("~show_delta", False))
        self.imu_extrinsic_file = str(
            rospy.get_param("~imu_extrinsic_file", str(default_extrinsic_path()))
        )
        self._imu_extrinsic: Optional[ImuExtrinsic] = load_imu_extrinsic(
            resolve_extrinsic_path(self.imu_extrinsic_file),
            required=False,
        )

        self.lock = threading.Lock()
        self.imu_buf = RollingRpyBuffer(self.window_sec)
        self.imu_calib_buf = RollingRpyBuffer(self.window_sec)
        self.mocap_buf = RollingRpyBuffer(self.window_sec)

        rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=200)
        rospy.Subscriber(self.mocap_topic, PoseStamped, self._mocap_cb, queue_size=100)

        nrows = 6 if self.show_delta else 3
        self.fig, axes = plt.subplots(nrows, 1, figsize=(12, 11 if nrows == 6 else 8), sharex=True)
        self.axes = np.atleast_1d(axes)

        title = f"IMU / IMU_calib / Mocap RPY (rolling {self.window_sec:.0f}s)"
        if self.show_delta:
            title += " + delta"
        self.fig.suptitle(title)

        unit = "deg" if self.plot_in_degrees else "rad"
        names = ["roll", "pitch", "yaw"]
        self.lines_imu: List = []
        self.lines_imu_calib: List = []
        self.lines_mocap: List = []
        self.lines_delta: List = []
        for i in range(3):
            ax = self.axes[i]
            line_m, = ax.plot([], [], "b-", linewidth=1.5, label="mocap")
            line_i, = ax.plot([], [], "r-", linewidth=1.5, label="imu")
            line_c, = ax.plot([], [], "g-", linewidth=1.5, label="imu_calib")
            self.lines_mocap.append(line_m)
            self.lines_imu.append(line_i)
            self.lines_imu_calib.append(line_c)
            ax.set_ylabel(f"{names[i]} [{unit}]")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper left")
        if self.show_delta:
            for j in range(3):
                ax = self.axes[3 + j]
                line_d, = ax.plot([], [], "m-", linewidth=1.2, label="imu - mocap")
                self.lines_delta.append(line_d)
                ax.set_ylabel(f"d{names[j]} [{unit}]")
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.legend(loc="upper left")
        self.axes[-1].set_xlabel("time [s], relative to now")
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.ion()
        plt.show()

        rospy.loginfo("plot_imu_mocap_rpy started.")
        rospy.loginfo("  imu=%s  mocap=%s  window=%.1fs  refresh=%.1fHz", self.imu_topic, self.mocap_topic, self.window_sec, self.refresh_hz)
        if self._imu_extrinsic is not None:
            rospy.loginfo(
                "  extrinsic loaded: %s (residual_rms=%.4f deg)",
                self.imu_extrinsic_file,
                self._imu_extrinsic.residual_angle_deg_rms,
            )
        else:
            rospy.logwarn("  no extrinsic at %s; imu_calib will be empty", self.imu_extrinsic_file)

    def _imu_cb(self, msg: Imu) -> None:
        quat = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            dtype=float,
        )
        rpy = quat_to_rpy(quat)
        if not np.all(np.isfinite(rpy)):
            return
        stamp = _stamp_sec(msg.header)
        with self.lock:
            self.imu_buf.append(stamp, rpy)
            if self._imu_extrinsic is not None:
                q_calib = self._imu_extrinsic.apply_quat(quat)
                if q_calib is not None:
                    rpy_calib = quat_to_rpy(q_calib)
                    if np.all(np.isfinite(rpy_calib)):
                        self.imu_calib_buf.append(stamp, rpy_calib)

    def _mocap_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        rpy = quat_to_rpy(np.array([q.x, q.y, q.z, q.w], dtype=float))
        if not np.all(np.isfinite(rpy)):
            return
        with self.lock:
            self.mocap_buf.append(_stamp_sec(msg.header), rpy)

    @staticmethod
    def _to_plot_units(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray, degrees: bool):
        if degrees:
            return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
        return roll, pitch, yaw

    @staticmethod
    def _set_ylim(ax, *arrays: np.ndarray, min_span: float) -> None:
        ys = [y for y in arrays if y.size > 0]
        if not ys:
            ax.set_ylim(-1.0, 1.0)
            return
        y_all = np.hstack(ys)
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        span = max(y_max - y_min, min_span)
        center = 0.5 * (y_min + y_max)
        half = 0.5 * span
        ax.set_ylim(center - half, center + half)

    def _log_mean_delta_throttled(
        self,
        t_i: np.ndarray,
        r_i: np.ndarray,
        p_i: np.ndarray,
        y_i: np.ndarray,
        t_m: np.ndarray,
        r_m: np.ndarray,
        p_m: np.ndarray,
        y_m: np.ndarray,
    ) -> None:
        if t_i.size < 5 or t_m.size < 5:
            return
        pick = nearest_indices(t_i, t_m)
        dr = wrap_angle_arr(r_i - r_m[pick])
        dp = wrap_angle_arr(p_i - p_m[pick])
        dy = wrap_angle_arr(y_i - y_m[pick])
        if self.plot_in_degrees:
            dr, dp, dy = np.rad2deg(dr), np.rad2deg(dp), np.rad2deg(dy)
        unit = "deg" if self.plot_in_degrees else "rad"
        rospy.loginfo_throttle(
            3.0,
            "mean delta (IMU-mocap): roll=%.2f pitch=%.2f yaw=%.2f %s",
            float(np.mean(dr)),
            float(np.mean(dp)),
            float(np.mean(dy)),
            unit,
        )

    def _log_mean_delta_calib_throttled(
        self,
        t_c: np.ndarray,
        r_c: np.ndarray,
        p_c: np.ndarray,
        y_c: np.ndarray,
        t_m: np.ndarray,
        r_m: np.ndarray,
        p_m: np.ndarray,
        y_m: np.ndarray,
    ) -> None:
        if t_c.size < 5 or t_m.size < 5:
            return
        pick = nearest_indices(t_c, t_m)
        dr = wrap_angle_arr(r_c - r_m[pick])
        dp = wrap_angle_arr(p_c - p_m[pick])
        dy = wrap_angle_arr(y_c - y_m[pick])
        if self.plot_in_degrees:
            dr, dp, dy = np.rad2deg(dr), np.rad2deg(dp), np.rad2deg(dy)
        unit = "deg" if self.plot_in_degrees else "rad"
        rospy.loginfo_throttle(
            3.0,
            "mean delta (IMU_calib-mocap): roll=%.2f pitch=%.2f yaw=%.2f %s",
            float(np.mean(dr)),
            float(np.mean(dp)),
            float(np.mean(dy)),
            unit,
        )

    def update_plot(self) -> None:
        now = rospy.Time.now().to_sec()
        with self.lock:
            self.imu_buf.trim(now)
            self.imu_calib_buf.trim(now)
            self.mocap_buf.trim(now)
            t_i, r_i, p_i, y_i = self.imu_buf.arrays(self.unwrap_angles)
            t_c, r_c, p_c, y_c = self.imu_calib_buf.arrays(self.unwrap_angles)
            t_m, r_m, p_m, y_m = self.mocap_buf.arrays(self.unwrap_angles)
            self._log_mean_delta_throttled(t_i, r_i, p_i, y_i, t_m, r_m, p_m, y_m)
            if self._imu_extrinsic is not None:
                self._log_mean_delta_calib_throttled(t_c, r_c, p_c, y_c, t_m, r_m, p_m, y_m)

        x_i = t_i - now if t_i.size else np.array([])
        x_c = t_c - now if t_c.size else np.array([])
        x_m = t_m - now if t_m.size else np.array([])

        r_i_p, p_i_p, y_i_p = self._to_plot_units(r_i, p_i, y_i, self.plot_in_degrees)
        r_c_p, p_c_p, y_c_p = self._to_plot_units(r_c, p_c, y_c, self.plot_in_degrees)
        r_m_p, p_m_p, y_m_p = self._to_plot_units(r_m, p_m, y_m, self.plot_in_degrees)
        rpy_imu = (r_i_p, p_i_p, y_i_p)
        rpy_imu_calib = (r_c_p, p_c_p, y_c_p)
        rpy_mocap = (r_m_p, p_m_p, y_m_p)
        min_span = 0.2 if self.plot_in_degrees else 0.02

        for i in range(3):
            self.lines_imu[i].set_data(x_i, rpy_imu[i])
            self.lines_imu_calib[i].set_data(x_c, rpy_imu_calib[i])
            self.lines_mocap[i].set_data(x_m, rpy_mocap[i])
            ax = self.axes[i]
            ax.set_xlim(-self.window_sec, 0.0)
            self._set_ylim(ax, rpy_imu[i], rpy_imu_calib[i], rpy_mocap[i], min_span=min_span)

        if self.show_delta and t_i.size and t_m.size:
            pick = nearest_indices(t_i, t_m)
            deltas_rad = (
                wrap_angle_arr(r_i - r_m[pick]),
                wrap_angle_arr(p_i - p_m[pick]),
                wrap_angle_arr(y_i - y_m[pick]),
            )
            if self.plot_in_degrees:
                deltas = tuple(np.rad2deg(d) for d in deltas_rad)
            else:
                deltas = deltas_rad
            for j, d in enumerate(deltas):
                self.lines_delta[j].set_data(x_i, d)
                ax = self.axes[3 + j]
                ax.set_xlim(-self.window_sec, 0.0)
                self._set_ylim(ax, d, min_span=min_span)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self) -> None:
        rate = rospy.Rate(max(self.refresh_hz, 1.0))
        while not rospy.is_shutdown():
            self.update_plot()
            plt.pause(0.001)
            rate.sleep()


def main() -> None:
    rospy.init_node("plot_imu_mocap_rpy", anonymous=False)
    node = ImuMocapRpyPlotNode()
    node.run()


if __name__ == "__main__":
    main()
