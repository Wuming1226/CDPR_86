#!/usr/bin/env python3
"""
Subscribe to /imu and estimate accelerometer / gyroscope noise std online.

The diagonal of EulerEKFCDPR.self.q (cdpr_euler_ekf.py) is process noise variance for:
  [w_a(3), w_g(3), w_ba(3), w_bg(3)]

From a stationary IMU, sample std of linear_acceleration and angular_velocity
estimates w_a and w_g standard deviations (variance = std^2 on the diagonal).
Bias random-walk terms w_ba / w_bg need long-term Allan analysis; this script
keeps EKF defaults unless you override via ROS params.

Usage:
  rosrun cdpr_86_host estimate_imu_process_noise.py
  rosrun cdpr_86_host estimate_imu_process_noise.py _imu_topic:=/imu _window_sec:=15
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import Imu


def _stamp_sec(msg: Imu) -> float:
    if msg.header.stamp != rospy.Time():
        return msg.header.stamp.to_sec()
    return rospy.Time.now().to_sec()


class RollingImuStats:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self._t: Deque[float] = deque()
        self._acc: Deque[np.ndarray] = deque()
        self._gyr: Deque[np.ndarray] = deque()

    def append(self, stamp_sec: float, acc: np.ndarray, gyr: np.ndarray) -> None:
        self._t.append(stamp_sec)
        self._acc.append(acc.copy())
        self._gyr.append(gyr.copy())
        self._trim(stamp_sec)

    def _trim(self, newest_sec: float) -> None:
        threshold = newest_sec - self.window_sec
        while self._t and self._t[0] < threshold:
            self._t.popleft()
            self._acc.popleft()
            self._gyr.popleft()

    @property
    def count(self) -> int:
        return len(self._t)

    def stats(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if self.count < 2:
            return None
        acc = np.vstack(self._acc)
        gyr = np.vstack(self._gyr)
        acc_mean = acc.mean(axis=0)
        gyr_mean = gyr.mean(axis=0)
        acc_std = acc.std(axis=0, ddof=1)
        gyr_std = gyr.std(axis=0, ddof=1)
        return acc_mean, acc_std, gyr_mean, gyr_std


def _fmt_vec(v: np.ndarray, unit: str = "") -> str:
    parts = [f"{x:.6g}" for x in v]
    s = "[" + ", ".join(parts) + "]"
    return s + (f" {unit}" if unit else "")


def _q_snippet(acc_std: float, gyr_std_rad: float, ba_std: float, bg_std_rad: float) -> str:
    """Match cdpr_euler_ekf.py: diagonal entries are variance = std**2."""
    bg_std_deg = math.degrees(bg_std_rad)
    gyr_std_deg = math.degrees(gyr_std_rad)
    return (
        "self.q = np.diag(\n"
        "    np.hstack(\n"
        "        [\n"
        f"            np.full(3, {acc_std:.6g}**2),  # w_a, σ≈{acc_std:.6g} m/s²\n"
        f"            np.full(3, np.deg2rad({gyr_std_deg:.6g}) ** 2),  # w_g, σ≈{gyr_std_deg:.6g} deg/s\n"
        f"            np.full(3, {ba_std:.6g}**2),  # w_ba (default)\n"
        f"            np.full(3, np.deg2rad({bg_std_deg:.6g}) ** 2),  # w_bg (default)\n"
        "        ]\n"
        "    )\n"
        ")"
    )


class EstimateImuProcessNoiseNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.imu_topic = rospy.get_param("~imu_topic", args.imu_topic)
        self.window_sec = float(rospy.get_param("~window_sec", args.window_sec))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", args.refresh_hz))
        self.min_samples = int(rospy.get_param("~min_samples", args.min_samples))

        # EKF defaults for bias random walk (variance = std^2)
        self.default_ba_std = float(rospy.get_param("~default_ba_std", 2e-4))
        self.default_bg_deg = float(rospy.get_param("~default_bg_deg", 0.02))

        self.buf = RollingImuStats(self.window_sec)
        self._msg_count = 0
        rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=500)

        rospy.loginfo(
            "estimate_imu_process_noise: topic=%s window=%.1fs refresh=%.1fHz min_samples=%d",
            self.imu_topic,
            self.window_sec,
            self.refresh_hz,
            self.min_samples,
        )
        rospy.loginfo("Keep the platform stationary for w_a / w_g estimates.")

    def _imu_cb(self, msg: Imu) -> None:
        acc = np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            dtype=float,
        )
        gyr = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=float,
        )
        self.buf.append(_stamp_sec(msg), acc, gyr)
        self._msg_count += 1

    def spin_report(self) -> None:
        rate = rospy.Rate(max(self.refresh_hz, 0.1))
        while not rospy.is_shutdown():
            out = self.buf.stats()
            n = self.buf.count
            if out is None or n < self.min_samples:
                rospy.loginfo_throttle(
                    2.0,
                    "Collecting IMU samples: %d / %d (need >= %d in %.1fs window)",
                    n,
                    self.min_samples,
                    self.min_samples,
                    self.window_sec,
                )
                rate.sleep()
                continue

            acc_mean, acc_std, gyr_mean, gyr_std = out
            acc_std_mean = float(np.mean(acc_std))
            gyr_std_mean = float(np.mean(gyr_std))

            ba_std = self.default_ba_std
            bg_std_rad = float(np.deg2rad(self.default_bg_deg))

            rospy.loginfo(
                "IMU noise (window %.1fs, n=%d, total msgs=%d)\n"
                "  accel mean (m/s²): %s\n"
                "  accel std  (m/s²): %s  -> w_a σ≈%.6g, var≈%.6g\n"
                "  gyro  mean (rad/s): %s\n"
                "  gyro  std  (rad/s): %s  -> w_g σ≈%.6g rad/s (%.4g deg/s), var≈%.6g\n"
                "  w_ba / w_bg: using defaults σ_ba=%.2e, σ_bg=%.4g deg/s\n"
                "--- paste into cdpr_euler_ekf.py EulerEKFCDPR.__init__ ---\n%s",
                self.window_sec,
                n,
                self._msg_count,
                _fmt_vec(acc_mean, "m/s²"),
                _fmt_vec(acc_std, "m/s²"),
                acc_std_mean,
                acc_std_mean**2,
                _fmt_vec(gyr_mean, "rad/s"),
                _fmt_vec(gyr_std, "rad/s"),
                gyr_std_mean,
                math.degrees(gyr_std_mean),
                gyr_std_mean**2,
                self.default_ba_std,
                self.default_bg_deg,
                _q_snippet(acc_std_mean, gyr_std_mean, ba_std, bg_std_rad),
            )
            rate.sleep()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate IMU process noise for EKF self.q from /imu.")
    p.add_argument("--imu-topic", default="/imu", help="IMU topic (overridden by ~imu_topic param).")
    p.add_argument("--window-sec", type=float, default=10.0, help="Rolling window length (seconds).")
    p.add_argument("--refresh-hz", type=float, default=2.0, help="How often to print estimates.")
    p.add_argument("--min-samples", type=int, default=50, help="Minimum samples before reporting.")
    return p.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("estimate_imu_process_noise", anonymous=False)
    args = _parse_args()
    node = EstimateImuProcessNoiseNode(args)
    node.spin_report()


if __name__ == "__main__":
    main()
