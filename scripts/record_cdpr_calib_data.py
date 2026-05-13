#!/usr/bin/env python3
"""
Record CDPR calibration samples on demand.

This node only records data. Move the CDPR and make sure all cables are taut
manually, then press Enter in this terminal. The script averages the following
window of pose/IMU/encoder data and appends one line to a txt file:

    x y z yaw pitch roll theta1 theta2 ... theta8

Default topics:
    /vrpn_client_node/cdpr/pose    geometry_msgs/PoseStamped
    /imu                          sensor_msgs/Imu
    /motor_pos_abs                std_msgs/Float32MultiArray

Stop with Ctrl-C.
"""

from collections import deque
import argparse
from datetime import datetime
import json
import math
import threading
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


N_CABLES = 8
COUNTS_PER_REV = 10000.0


def circular_mean(angles: np.ndarray) -> float:
    """
    Mean angle in radians.

    Directly averaging angles fails around the -pi/pi boundary. Averaging sin
    and cos first gives the correct circular mean for yaw/pitch/roll windows.
    """
    return math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))


def mean_ypr(ypr_values: np.ndarray) -> np.ndarray:
    """Average yaw, pitch, roll columns independently with circular means."""
    return np.array([circular_mean(ypr_values[:, i]) for i in range(3)], dtype=float)


def trim_buffer(buf: Deque[Tuple[float, np.ndarray]], newest_stamp: float, keep_sec: float) -> None:
    """Keep only samples that may still be used by a future averaging window."""
    oldest_allowed = newest_stamp - keep_sec
    while buf and buf[0][0] < oldest_allowed:
        buf.popleft()


def values_in_window(buf: Deque[Tuple[float, np.ndarray]], start: float, end: float) -> np.ndarray:
    """Return all buffered values whose timestamps fall in [start, end]."""
    values = [value for stamp, value in buf if start <= stamp <= end]
    if not values:
        return np.empty((0, 0), dtype=float)
    return np.vstack(values)


class CDPRCalibrationRecorder:
    def __init__(
        self,
        output_path: Path,
        window_sec: float,
        pose_topic: str,
        imu_topic: str,
        motor_abs_topic: str,
        init_motor_window_sec: float,
        angle_degrees: bool,
    ):
        self.output_path = output_path
        self.window_sec = float(window_sec)
        self.init_motor_window_sec = float(init_motor_window_sec)
        self.keep_sec = max(2.0 * self.window_sec, self.window_sec + 1.0, self.init_motor_window_sec + 1.0)
        self.angle_degrees = bool(angle_degrees)

        self.lock = threading.Lock()
        self.pose_buf: Deque[Tuple[float, np.ndarray]] = deque()
        self.imu_buf: Deque[Tuple[float, np.ndarray]] = deque()
        self.motor_abs_buf: Deque[Tuple[float, np.ndarray]] = deque()
        self.init_motor_pos_abs: np.ndarray = np.zeros(N_CABLES, dtype=float)

        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=200)
        rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=200)
        rospy.Subscriber(motor_abs_topic, Float32MultiArray, self.motor_abs_callback, queue_size=200)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def pose_callback(self, msg: PoseStamped) -> None:
        stamp = self.message_time(msg.header.stamp)
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        with self.lock:
            self.pose_buf.append((stamp, pos))
            trim_buffer(self.pose_buf, stamp, self.keep_sec)

    def imu_callback(self, msg: Imu) -> None:
        stamp = self.message_time(msg.header.stamp)
        quat = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-12:
            return

        # SciPy uppercase "ZYX" corresponds to intrinsic ZYX Euler angles.
        # It returns angles in the same order requested: yaw, pitch, roll.
        ypr = R.from_quat(quat).as_euler("ZYX", degrees=False)
        with self.lock:
            self.imu_buf.append((stamp, ypr))
            trim_buffer(self.imu_buf, stamp, self.keep_sec)

    def motor_abs_callback(self, msg: Float32MultiArray) -> None:
        stamp = rospy.Time.now().to_sec()
        theta = np.asarray(msg.data, dtype=float).reshape(-1)
        if theta.size < N_CABLES:
            rospy.logwarn_throttle(2.0, "motor_pos_abs has %d values, expected at least %d.", theta.size, N_CABLES)
            return
        theta = theta[:N_CABLES]
        with self.lock:
            self.motor_abs_buf.append((stamp, theta))
            trim_buffer(self.motor_abs_buf, stamp, self.keep_sec)

    @staticmethod
    def message_time(stamp: rospy.Time) -> float:
        """Use message header time when available; otherwise fall back to ROS now."""
        if stamp is not None and stamp.to_sec() > 0.0:
            return stamp.to_sec()
        return rospy.Time.now().to_sec()

    def wait_for_initial_data(self) -> None:
        """Block until every topic has produced at least one message."""
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            with self.lock:
                ready = bool(self.pose_buf and self.imu_buf and self.motor_abs_buf)
            if ready:
                return
            rospy.loginfo_throttle(2.0, "Waiting for pose, imu and motor_pos_abs messages...")
            rate.sleep()

    def record_initial_motor_pos_abs(self) -> np.ndarray:
        """Average motor_pos_abs at startup and store it as file metadata."""
        start = rospy.Time.now().to_sec()
        end = start + self.init_motor_window_sec
        rospy.loginfo("Averaging motor_pos_abs for %.2f s...", self.init_motor_window_sec)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < end:
            rospy.sleep(0.01)

        end = rospy.Time.now().to_sec()
        with self.lock:
            motor_abs_values = values_in_window(self.motor_abs_buf, start, end)

        if motor_abs_values.size == 0:
            raise RuntimeError("No motor_pos_abs data in initial averaging window.")

        init_motor_pos_abs = np.mean(motor_abs_values, axis=0)
        rospy.loginfo("Initial motor_pos_abs averaged from %d samples.", motor_abs_values.shape[0])
        return init_motor_pos_abs

    def write_header_if_needed(self, init_motor_pos_abs: np.ndarray) -> None:
        """Create a new txt file with metadata first, then the column header."""
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            rospy.logwarn(
                "%s already exists; keeping existing header and appending samples. "
                "If you need a new init_motor_pos_abs metadata line, record to a new file.",
                self.output_path,
            )
            return

        metadata = {
            "init_motor_pos_abs": init_motor_pos_abs.tolist(),
            "init_motor_pos_abs_window_sec": self.init_motor_window_sec,
            "angle_unit": "deg" if self.angle_degrees else "rad",
            "theta_definition": "(motor_pos_abs - init_motor_pos_abs) / 10000 * 2*pi",
            "counts_per_revolution": COUNTS_PER_REV,
            "init_motor_topic": "motor_pos_abs",
        }
        with self.output_path.open("w", encoding="utf-8") as f:
            f.write(f"# cdpr_calib_metadata {json.dumps(metadata, separators=(',', ':'))}\n")
            f.write("# x y z yaw pitch roll theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8\n")

    def record_once(self) -> bool:
        """
        Average the window after Enter is pressed and append one calibration row.

        Returns False when one of the three topics does not have enough data in
        the requested window; the user can keep the CDPR still and press Enter again.
        """
        start = rospy.Time.now().to_sec()
        end = start + self.window_sec
        rospy.loginfo("Recording next %.2f s window...", self.window_sec)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < end:
            rospy.sleep(0.01)

        end = rospy.Time.now().to_sec()
        with self.lock:
            pose_values = values_in_window(self.pose_buf, start, end)
            imu_values = values_in_window(self.imu_buf, start, end)
            motor_abs_values = values_in_window(self.motor_abs_buf, start, end)

        if pose_values.size == 0 or imu_values.size == 0 or motor_abs_values.size == 0:
            rospy.logwarn(
                "Not enough data after Enter in %.2f s window: pose=%d imu=%d motor_abs=%d",
                self.window_sec,
                pose_values.shape[0],
                imu_values.shape[0],
                motor_abs_values.shape[0],
            )
            return False

        pos_mean = np.mean(pose_values, axis=0)
        ypr_mean = mean_ypr(imu_values)
        motor_abs_mean = np.mean(motor_abs_values, axis=0)
        theta_mean = (motor_abs_mean - self.init_motor_pos_abs) / COUNTS_PER_REV * (2.0 * math.pi)
        if self.angle_degrees:
            ypr_out = np.rad2deg(ypr_mean)
        else:
            ypr_out = ypr_mean

        row = np.hstack([pos_mean, ypr_out, theta_mean])
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(" ".join(f"{v:.10g}" for v in row))
            f.write("\n")

        rospy.loginfo(
            "Recorded sample: pose=%d imu=%d motor_abs=%d -> %s",
            pose_values.shape[0],
            imu_values.shape[0],
            motor_abs_values.shape[0],
            self.output_path,
        )
        return True

    def run(self) -> None:
        self.wait_for_initial_data()
        self.init_motor_pos_abs = self.record_initial_motor_pos_abs()
        self.write_header_if_needed(self.init_motor_pos_abs)
        rospy.loginfo(
            "Ready. Move CDPR, ensure all cables are taut, then press Enter to record a %.2f s average. Ctrl-C to stop.",
            self.window_sec,
        )
        while not rospy.is_shutdown():
            try:
                input()
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            self.record_once()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record CDPR calibration data by pressing Enter.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cdpr_calibration_samples.txt"),
        help="Base output path; a timestamp _MMDDHHMM is inserted before .txt (e.g. ..._05131913.txt).",
    )
    parser.add_argument("--window", type=float, default=2.0, help="Averaging window width in seconds.")
    parser.add_argument("--pose-topic", default="/vrpn_client_node/cdpr/pose", help="Mocap pose topic.")
    parser.add_argument("--imu-topic", default="/imu", help="IMU topic.")
    parser.add_argument("--motor-abs-topic", default="/motor_pos_abs", help="Absolute motor encoder topic for init metadata.")
    parser.add_argument("--init-motor-window", type=float, default=2.0, help="Initial motor_pos_abs averaging window in seconds.")
    parser.add_argument(
        "--angle-degrees",
        action="store_true",
        help="Write yaw/pitch/roll in degrees. Default is radians for calibrate_cdpr_kinematics.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%m%d%H%M")
    args.output = args.output.with_name(f"{args.output.stem}_{ts}{args.output.suffix}")
    rospy.init_node("record_cdpr_calib_data", anonymous=False)
    recorder = CDPRCalibrationRecorder(
        output_path=args.output,
        window_sec=args.window,
        pose_topic=args.pose_topic,
        imu_topic=args.imu_topic,
        motor_abs_topic=args.motor_abs_topic,
        init_motor_window_sec=args.init_motor_window,
        angle_degrees=args.angle_degrees,
    )
    recorder.run()


if __name__ == "__main__":
    main()
