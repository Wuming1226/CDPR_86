#!/usr/bin/env python3
"""
Record IMU / mocap orientation pairs for extrinsic calibration.

Move the CDPR end-effector to a pose, keep cables taut and hold still, press Enter.
The script averages the next ``window_sec`` seconds of orientations and appends one line:

    q_imu_x q_imu_y q_imu_z q_imu_w q_mocap_x q_mocap_y q_mocap_z q_mocap_w

Default output (no timestamp suffix): scripts/cdpr_imu_extrinsic_samples.txt

Then run:
    python3 calibrate_imu_extrinsic.py
    python3 calibrate_imu_extrinsic.py --rows "0:5,7"   # optional row subset (same as kinematic calib)
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu

DEFAULT_SAMPLES_FILE = "cdpr_imu_extrinsic_samples.txt"


def trim_buffer(buf: Deque[Tuple[float, np.ndarray]], newest_stamp: float, keep_sec: float) -> None:
    oldest_allowed = newest_stamp - keep_sec
    while buf and buf[0][0] < oldest_allowed:
        buf.popleft()


def values_in_window(buf: Deque[Tuple[float, np.ndarray]], start: float, end: float) -> List[np.ndarray]:
    return [value for stamp, value in buf if start <= stamp <= end]


def mean_quat(quats: List[np.ndarray]) -> np.ndarray:
    rots = []
    for q in quats:
        q = np.asarray(q, dtype=float).reshape(4)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
            continue
        rots.append(R.from_quat(q / np.linalg.norm(q)))
    if not rots:
        raise ValueError("No valid quaternions in window")
    return R.concatenate(rots).mean().as_quat()


class ImuExtrinsicRecorder:
    def __init__(
        self,
        output_path: Path,
        window_sec: float,
        pose_topic: str,
        imu_topic: str,
    ) -> None:
        self.output_path = output_path
        self.window_sec = float(window_sec)
        self.keep_sec = max(2.0 * self.window_sec, self.window_sec + 1.0)

        self.lock = threading.Lock()
        self.imu_buf: Deque[Tuple[float, np.ndarray]] = deque()
        self.mocap_buf: Deque[Tuple[float, np.ndarray]] = deque()

        rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=300)
        rospy.Subscriber(pose_topic, PoseStamped, self.mocap_callback, queue_size=200)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def message_time(stamp: rospy.Time) -> float:
        if stamp != rospy.Time():
            return stamp.to_sec()
        return rospy.Time.now().to_sec()

    @staticmethod
    def _quat_from_imu(msg: Imu) -> np.ndarray:
        return np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            dtype=float,
        )

    @staticmethod
    def _quat_from_pose(msg: PoseStamped) -> np.ndarray:
        q = msg.pose.orientation
        return np.array([q.x, q.y, q.z, q.w], dtype=float)

    def imu_callback(self, msg: Imu) -> None:
        stamp = self.message_time(msg.header.stamp)
        quat = self._quat_from_imu(msg)
        if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-12:
            return
        with self.lock:
            self.imu_buf.append((stamp, quat))
            trim_buffer(self.imu_buf, stamp, self.keep_sec)

    def mocap_callback(self, msg: PoseStamped) -> None:
        stamp = self.message_time(msg.header.stamp)
        quat = self._quat_from_pose(msg)
        if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-12:
            return
        with self.lock:
            self.mocap_buf.append((stamp, quat))
            trim_buffer(self.mocap_buf, stamp, self.keep_sec)

    def write_header_if_needed(self) -> None:
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            rospy.logwarn(
                "%s already exists; appending samples. Delete the file to start fresh.",
                self.output_path,
            )
            return
        meta = {
            "window_sec": self.window_sec,
            "convention": "scipy quat xyzw; R_world_body = R_world_imu @ R_imu_to_body",
        }
        with self.output_path.open("w", encoding="utf-8") as f:
            f.write(f"# imu_extrinsic_metadata {json.dumps(meta, separators=(',', ':'))}\n")
            f.write("# q_imu_x q_imu_y q_imu_z q_imu_w q_mocap_x q_mocap_y q_mocap_z q_mocap_w\n")

    def wait_for_initial_data(self) -> None:
        rate = rospy.Rate(20.0)
        rospy.loginfo("Waiting for IMU and mocap messages...")
        while not rospy.is_shutdown():
            with self.lock:
                ready = bool(self.imu_buf and self.mocap_buf)
            if ready:
                rospy.loginfo("IMU and mocap data ready.")
                return
            rate.sleep()

    def record_once(self) -> bool:
        start = rospy.Time.now().to_sec()
        end = start + self.window_sec
        rospy.loginfo("Recording next %.2f s window...", self.window_sec)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < end:
            rospy.sleep(0.01)

        end = rospy.Time.now().to_sec()
        with self.lock:
            imu_quats = values_in_window(self.imu_buf, start, end)
            mocap_quats = values_in_window(self.mocap_buf, start, end)

        if len(imu_quats) < 5 or len(mocap_quats) < 5:
            rospy.logwarn(
                "Not enough data in %.2f s window: imu=%d mocap=%d (need >= 5 each). Hold still and press Enter again.",
                self.window_sec,
                len(imu_quats),
                len(mocap_quats),
            )
            return False

        try:
            q_imu = mean_quat(imu_quats)
            q_mocap = mean_quat(mocap_quats)
        except ValueError as exc:
            rospy.logwarn("Failed to average quaternions: %s", exc)
            return False

        row = np.hstack([q_imu, q_mocap])
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(" ".join(f"{v:.12g}" for v in row))
            f.write("\n")

        yaw_i, pitch_i, roll_i = R.from_quat(q_imu).as_euler("ZYX", degrees=True)
        yaw_m, pitch_m, roll_m = R.from_quat(q_mocap).as_euler("ZYX", degrees=True)
        rospy.loginfo(
            "Recorded pose #%d: imu_n=%d mocap_n=%d -> %s",
            self._count_lines(),
            len(imu_quats),
            len(mocap_quats),
            self.output_path,
        )
        rospy.loginfo(
            "  IMU  rpy [deg]: roll=%.2f pitch=%.2f yaw=%.2f",
            roll_i,
            pitch_i,
            yaw_i,
        )
        rospy.loginfo(
            "  Mocap rpy [deg]: roll=%.2f pitch=%.2f yaw=%.2f",
            roll_m,
            pitch_m,
            yaw_m,
        )
        return True

    def _count_lines(self) -> int:
        if not self.output_path.is_file():
            return 0
        n = 0
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    n += 1
        return n

    def run(self) -> None:
        self.wait_for_initial_data()
        self.write_header_if_needed()
        rospy.loginfo(
            "Ready. Move CDPR to a pose, hold still, press Enter to record %.2f s average. Ctrl-C to stop.",
            self.window_sec,
        )
        rospy.loginfo("Output file: %s", self.output_path.resolve())
        while not rospy.is_shutdown():
            try:
                input()
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            self.record_once()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record IMU/mocap quaternion pairs for extrinsic calibration.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_SAMPLES_FILE),
        help="Output txt path (default: cdpr_imu_extrinsic_samples.txt, no timestamp suffix).",
    )
    p.add_argument("--window", type=float, default=2.0, help="Averaging window after Enter [s].")
    p.add_argument("--pose-topic", default="/vrpn_client_node/cdpr/pose")
    p.add_argument("--imu-topic", default="/imu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("record_imu_extrinsic_data", anonymous=False)
    out = args.output
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    recorder = ImuExtrinsicRecorder(
        output_path=out,
        window_sec=float(rospy.get_param("~window_sec", args.window)),
        pose_topic=rospy.get_param("~mocap_topic", args.pose_topic),
        imu_topic=rospy.get_param("~imu_topic", args.imu_topic),
    )
    recorder.run()


if __name__ == "__main__":
    main()
