#!/usr/bin/env python3
"""
Live plot: 8 cable lengths from moving-platform pose + fixed anchors (geometry)
vs lengths from motor encoders (same model as cable_lengths_measure).

Layout: 4x2 subplots, 10 s rolling window on the time axis (relative to now).
"""

from collections import deque
import threading

import matplotlib.pyplot as plt
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from cdpr import CDPR


class RollingBuffer:
    def __init__(self, window_sec: float, dim: int):
        self.window_sec = float(window_sec)
        self.dim = int(dim)
        self.t = deque()
        self.values = [deque() for _ in range(self.dim)]

    def append(self, stamp_sec: float, vec: np.ndarray) -> None:
        self.t.append(stamp_sec)
        for i in range(self.dim):
            self.values[i].append(float(vec[i]))
        self.trim(stamp_sec)

    def trim(self, newest_sec: float) -> None:
        threshold = newest_sec - self.window_sec
        while self.t and self.t[0] < threshold:
            self.t.popleft()
            for i in range(self.dim):
                self.values[i].popleft()


class CablePoseVsEncoderPlot:
    def __init__(self):
        rospy.init_node("plot_cable_pose_vs_encoder", anonymous=False)
        self.window_sec = float(rospy.get_param("~window_sec", 10.0))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 20.0))
        imu_active = bool(rospy.get_param("~imu_active", False))
        imu_topic = rospy.get_param("~imu_topic", "/imu")

        self.lock = threading.Lock()
        self.buf_geom = RollingBuffer(self.window_sec, dim=8)
        self.buf_enc = RollingBuffer(self.window_sec, dim=8)

        rospy.loginfo("Initializing CDPR (waits for motor_pos_rel)...")
        self.cdpr = CDPR(imu_active=imu_active, imu_topic=imu_topic)
        rospy.loginfo("CDPR ready.")

        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        self.axes = self.axes.flatten()
        self.fig.suptitle(
            "Cable length: geometry (mocap pose + anchors) vs encoder (10 s window)"
        )
        self.lines_geom = []
        self.lines_enc = []
        for i, ax in enumerate(self.axes):
            lg, = ax.plot([], [], "b-", linewidth=1.4, label="geom")
            le, = ax.plot([], [], "r--", linewidth=1.2, label="encoder")
            self.lines_geom.append(lg)
            self.lines_enc.append(le)
            ax.set_ylabel(f"L{i + 1} [m]")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper left", fontsize=8)
        self.axes[-2].set_xlabel("time [s] (relative to now)")
        self.axes[-1].set_xlabel("time [s] (relative to now)")
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.ion()
        plt.show()
        rospy.loginfo("plot_cable_pose_vs_encoder: window=%.1fs refresh=%.1f Hz", self.window_sec, self.refresh_hz)

    def sample(self) -> None:
        now = rospy.Time.now().to_sec()
        x, y, z, quat = self.cdpr.get_moving_platform_pose_from_mocap()
        pos = np.array([x, y, z], dtype=float)
        rot = R.from_quat(quat)
        L_geom = self.cdpr.calculate_cable_length_at_pose(pos, rot)
        L_enc = self.cdpr.calculate_cable_length_from_motor_pos(self.cdpr.motor_pos)
        with self.lock:
            self.buf_geom.append(now, L_geom)
            self.buf_enc.append(now, L_enc)

    def update_plot(self) -> None:
        now = rospy.Time.now().to_sec()
        with self.lock:
            self.buf_geom.trim(now)
            self.buf_enc.trim(now)
            t = np.array(self.buf_geom.t, dtype=float)
            xrel = t - now if t.size > 0 else np.array([])
            for i in range(8):
                yg = np.array(self.buf_geom.values[i], dtype=float)
                ye = np.array(self.buf_enc.values[i], dtype=float)
                self.lines_geom[i].set_data(xrel, yg)
                self.lines_enc[i].set_data(xrel, ye)
                ax = self.axes[i]
                ax.set_xlim(-self.window_sec, 0.0)
                min_span_m = 0.1
                if yg.size > 0 or ye.size > 0:
                    y_all = np.hstack([yg, ye]) if yg.size and ye.size else (yg if yg.size else ye)
                    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
                else:
                    y_min, y_max = 0.0, min_span_m
                center = 0.5 * (y_min + y_max)
                span = max(y_max - y_min, min_span_m)
                half = 0.5 * span
                ax.set_ylim(center - half, center + half)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self) -> None:
        rate = rospy.Rate(self.refresh_hz)
        while not rospy.is_shutdown():
            self.sample()
            self.update_plot()
            plt.pause(0.001)
            rate.sleep()


def main():
    node = CablePoseVsEncoderPlot()
    node.run()


if __name__ == "__main__":
    main()
