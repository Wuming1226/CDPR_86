#!/usr/bin/env python3
"""
Record IMU + FK pose topic + RealSense D455 color video with aligned timestamps.

Outputs a session folder:
  - video.mp4
  - video_frames.csv
  - imu.csv
  - mag.csv         (magnetometer from /yesense/inertial_data)
  - fk.csv          (x,y,z,qx,qy,qz,qw)
  - mocap.csv       (x,y,z,qx,qy,qz,qw)
  - motor_pos_abs.csv
  - fk_mocap_compare.png (FK vs mocap pose comparison)
  - metadata.json

Time alignment strategy:
  - Main timeline uses system epoch seconds: time.time()
  - ROS messages keep their header.stamp.to_sec() (when available)
  - All recorded timestamps in CSV are in seconds (s).
  - RealSense color frames are mapped to system epoch seconds using either:
      * SYSTEM_TIME domain: t_frame_sys = frame_ts (SDK ms converted to s)
      * otherwise: t_frame_sys = t_sys0 + (frame_ts - t_cam0) (first-frame alignment, s)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# Global switch: set False to disable D455 capture and video saving.
ENABLE_CAMERA_RECORDING = True

# ROS subscriber queue depth for high-rate pose topics.
POSE_SUBSCRIBER_QUEUE_SIZE = 5000


def _import_rs_cv():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing `pyrealsense2`. Install: python3 -m pip install pyrealsense2") from e

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing `opencv-python`. Install: python3 -m pip install opencv-python") from e

    import numpy as np  # type: ignore

    return rs, cv2, np


def _import_ros():
    try:
        import rospy  # type: ignore
        from sensor_msgs.msg import Imu  # type: ignore
        from geometry_msgs.msg import PoseStamped  # type: ignore
        from cdpr_86_msg.msg import MotorPositionsStamped  # type: ignore
        from yesense_imu.msg import YesenseImuInertialData  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ROS python environment is not available.\n"
            "Fix by sourcing your ROS + workspace setup, e.g.:\n"
            "  source /opt/ros/<distro>/setup.bash\n"
            "  source <your_catkin_ws>/devel/setup.bash\n"
            "Then run this script again.\n"
            "If you only want to record video (no ROS), add: --video-only"
        ) from e

    return rospy, Imu, PoseStamped, MotorPositionsStamped, YesenseImuInertialData


def _timestamp_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ros_time_to_sec(stamp, rospy) -> float:
    try:
        if stamp is not None and stamp.to_sec() > 0.0:
            return float(stamp.to_sec())
    except Exception:
        pass
    return float(rospy.Time.now().to_sec())


@dataclass
class RosSysClockSync:
    ros0: float
    sys0: float

    def ros_to_sys(self, t_ros: float) -> float:
        # Assumes ROS wall-clock time (not /use_sim_time). Still useful as an estimate for logging.
        return self.sys0 + (t_ros - self.ros0)


class CsvWriterThread:
    def __init__(self, path: Path, header: list[str]):
        self.path = path
        self.header = header
        self.q: "queue.Queue[list[Any]]" = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name=f"csv_writer:{path.name}", daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
        self._thr.start()

    def put(self, row: list[Any]) -> None:
        if self._stop.is_set():
            return
        try:
            self.q.put_nowait(row)
        except queue.Full:
            # Drop rows rather than blocking real-time threads.
            pass

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thr.join(timeout=timeout)

    def _run(self) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            last_flush = time.time()
            while not self._stop.is_set() or not self.q.empty():
                try:
                    row = self.q.get(timeout=0.05)
                except queue.Empty:
                    row = None
                if row is not None:
                    w.writerow(row)
                now = time.time()
                if now - last_flush > 0.5:
                    f.flush()
                    last_flush = now
            f.flush()


POSE_CSV_HEADER = ["t_ros", "t_sys_est", "x", "y", "z", "qx", "qy", "qz", "qw"]
MOTOR_POS_CSV_HEADER = ["t_ros", "t_sys_est", "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]
MOTOR_POS_COUNT = 8


def _pose_msg_to_csv_row(msg, t_ros: float, t_sys_est: float) -> list[str]:
    """PoseStamped -> CSV row with position + quaternion (no RPY conversion)."""
    p = msg.pose.position
    q = msg.pose.orientation
    return [
        f"{t_ros:.9f}",
        f"{t_sys_est:.9f}",
        f"{p.x:.10g}",
        f"{p.y:.10g}",
        f"{p.z:.10g}",
        f"{q.x:.10g}",
        f"{q.y:.10g}",
        f"{q.z:.10g}",
        f"{q.w:.10g}",
    ]


def _motor_pos_msg_to_csv_row(msg, t_ros: float, t_sys_est: float) -> list[str]:
    row = [f"{t_ros:.9f}", f"{t_sys_est:.9f}"]
    for i in range(MOTOR_POS_COUNT):
        if i < len(msg.positions):
            row.append(f"{float(msg.positions[i]):.10g}")
        else:
            row.append("")
    return row


def _load_t_ros_series(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [row["t_ros"] for row in csv.DictReader(f)]


def compare_fk_motor_stamp_duplicates(session_dir: Path) -> dict[str, Any]:
    """Compare duplicate t_ros patterns between fk.csv and motor_pos_abs.csv."""
    fk_path = session_dir / "fk.csv"
    motor_path = session_dir / "motor_pos_abs.csv"
    if not fk_path.is_file() or not motor_path.is_file():
        print("[WARN] compare_fk_motor_stamp_duplicates: missing fk.csv or motor_pos_abs.csv", file=sys.stderr)
        return {}

    from collections import Counter

    fk_ts = _load_t_ros_series(fk_path)
    motor_ts = _load_t_ros_series(motor_path)
    fk_counts = Counter(fk_ts)
    motor_counts = Counter(motor_ts)

    fk_repeated = {t for t, c in fk_counts.items() if c > 1}
    motor_repeated = {t for t, c in motor_counts.items() if c > 1}
    both_repeated = fk_repeated & motor_repeated
    fk_only_repeated = fk_repeated - motor_repeated
    motor_only_repeated = motor_repeated - fk_repeated

    fk_adjacent = sum(1 for i in range(1, len(fk_ts)) if fk_ts[i] == fk_ts[i - 1])
    motor_adjacent = sum(1 for i in range(1, len(motor_ts)) if motor_ts[i] == motor_ts[i - 1])

    fk_adj_ts = {fk_ts[i] for i in range(1, len(fk_ts)) if fk_ts[i] == fk_ts[i - 1]}
    motor_adj_ts = {motor_ts[i] for i in range(1, len(motor_ts)) if motor_ts[i] == motor_ts[i - 1]}
    both_adjacent_ts = fk_adj_ts & motor_adj_ts
    fk_only_adjacent_ts = fk_adj_ts - motor_adj_ts

    overlap_ratio = (len(both_repeated) / len(fk_repeated)) if fk_repeated else 1.0
    adj_overlap_ratio = (len(both_adjacent_ts) / len(fk_adj_ts)) if fk_adj_ts else 1.0

    result = {
        "fk_rows": len(fk_ts),
        "motor_rows": len(motor_ts),
        "fk_repeated_timestamps": len(fk_repeated),
        "motor_repeated_timestamps": len(motor_repeated),
        "both_repeated_timestamps": len(both_repeated),
        "fk_only_repeated_timestamps": len(fk_only_repeated),
        "motor_only_repeated_timestamps": len(motor_only_repeated),
        "fk_adjacent_duplicate_pairs": fk_adjacent,
        "motor_adjacent_duplicate_pairs": motor_adjacent,
        "fk_adjacent_duplicate_timestamp_values": len(fk_adj_ts),
        "motor_adjacent_duplicate_timestamp_values": len(motor_adj_ts),
        "both_adjacent_duplicate_timestamp_values": len(both_adjacent_ts),
        "fk_only_adjacent_duplicate_timestamp_values": len(fk_only_adjacent_ts),
        "repeated_timestamp_overlap_ratio": overlap_ratio,
        "adjacent_timestamp_overlap_ratio": adj_overlap_ratio,
    }

    print("[INFO] FK vs motor_pos_abs duplicate timestamp comparison:", file=sys.stderr)
    for k, v in result.items():
        print(f"  {k}: {v}", file=sys.stderr)
    if fk_only_repeated:
        sample = sorted(fk_only_repeated)[:5]
        print(f"  fk_only_repeated_samples: {sample}", file=sys.stderr)
    if fk_only_adjacent_ts:
        sample = sorted(fk_only_adjacent_ts)[:5]
        print(f"  fk_only_adjacent_samples: {sample}", file=sys.stderr)

    return result


def plot_fk_mocap_from_csv(session_dir: Path, output_png: Optional[Path] = None) -> Optional[Path]:
    """Plot FK vs mocap from saved CSV files (post-recording)."""
    fk_path = session_dir / "fk.csv"
    mocap_path = session_dir / "mocap.csv"
    if not fk_path.is_file() or not mocap_path.is_file():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def _load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
        rows: list[tuple[float, list[float], list[float], bool]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["t_sys_est"])
                xyz = [float(row[c]) for c in ("x", "y", "z")]
                if all(c in row for c in ("qx", "qy", "qz", "qw")):
                    quat = [float(row[c]) for c in ("qx", "qy", "qz", "qw")]
                    rows.append((t, xyz, quat, False))
                elif all(c in row for c in ("roll", "pitch", "yaw")):
                    rpy = [float(row[c]) for c in ("roll", "pitch", "yaw")]
                    rows.append((t, xyz, rpy, True))
        if not rows:
            return np.array([]), np.empty((0, 6))

        t_arr = np.array([r[0] for r in rows], dtype=float)
        xyz_arr = np.array([r[1] for r in rows], dtype=float)
        if rows[0][3]:
            rpy_arr = np.array([r[2] for r in rows], dtype=float)
        else:
            quat_arr = np.array([r[2] for r in rows], dtype=float)
            rpy_arr = R.from_quat(quat_arr).as_euler("ZYX", degrees=False)
        pose_arr = np.column_stack([xyz_arr, rpy_arr])

        t0 = float(t_arr[0])
        return t_arr - t0, pose_arr

    t_fk, pose_fk = _load_pose_csv(fk_path)
    t_m, pose_m = _load_pose_csv(mocap_path)
    if t_fk.size == 0 or t_m.size == 0:
        return None

    labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("FK vs Mocap (full session)")
    for i, ax in enumerate(axes):
        y_m = pose_m[:, i]
        y_f = pose_fk[:, i]
        if i >= 3:
            y_m = np.rad2deg(y_m)
            y_f = np.rad2deg(y_f)
        ax.plot(t_m, y_m, "b-", linewidth=1.2, label="mocap")
        ax.plot(t_fk, y_f, "m-", linewidth=1.0, label="fk")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s] since session start (t_sys_est)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = output_png or (session_dir / "fk_mocap_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


class D455VideoRecorder:
    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        max_color: bool,
        show_video: bool,
        output_mp4: Path,
        output_frames_csv: Path,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.max_color = bool(max_color)
        self.show_video = bool(show_video)
        self.output_mp4 = output_mp4
        self.output_frames_csv = output_frames_csv
        self.window_name = "D455 Live"

        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name="d455_recorder", daemon=True)

        self._rs, self._cv2, self._np = _import_rs_cv()
        self._pipeline = self._rs.pipeline()
        self._config = self._rs.config()

        self._writer = None
        self._frame_writer = CsvWriterThread(
            self.output_frames_csv,
            header=[
                "frame_idx",
                "t_frame_sys",
                "t_arrival_sys",
                "frame_ts",
                "ts_domain",
                "width",
                "height",
                "fps",
            ],
        )

        self._t_sys0: Optional[float] = None
        self._t_cam0_s: Optional[float] = None
        self._recording_enabled = False
        self._frame_idx = 0

    def begin_recording(self) -> None:
        """Start writing video frames after warmup."""
        self._recording_enabled = True
        self._frame_idx = 0

    @staticmethod
    def _list_devices(rs) -> list[str]:
        serials: list[str] = []
        for dev in rs.context().query_devices():
            try:
                serials.append(dev.get_info(rs.camera_info.serial_number))
            except Exception:
                continue
        return serials

    @staticmethod
    def _select_device(rs, serial: Optional[str]):
        devices = rs.context().query_devices()
        if devices.size() == 0:
            raise RuntimeError("No RealSense device detected. Plug in D455 and retry.")
        if serial is None:
            return devices[0]
        for dev in devices:
            try:
                if dev.get_info(rs.camera_info.serial_number) == serial:
                    return dev
            except Exception:
                continue
        available = D455VideoRecorder._list_devices(rs)
        raise RuntimeError(f"RealSense serial={serial} not found. Available: {available}")

    @staticmethod
    def _pick_max_color_profile(rs, device, fps: int) -> Tuple[int, int, int]:
        # Same heuristic as scripts/record_d455.py
        color_sensor = None
        for s in device.sensors:
            try:
                profiles = s.get_stream_profiles()
                has_color = any(p.stream_type() == rs.stream.color for p in profiles)
                if has_color:
                    color_sensor = s
                    break
            except Exception:
                continue
        if color_sensor is None:
            raise RuntimeError("No color sensor found on this RealSense device.")

        candidates: list[Tuple[int, int, int]] = []
        for p in color_sensor.get_stream_profiles():
            try:
                if p.stream_type() != rs.stream.color or not p.is_video_stream_profile():
                    continue
                vp = p.as_video_stream_profile()
                w, h = int(vp.width()), int(vp.height())
                pfps = int(vp.fps())
                fmt = vp.format()
                if pfps != int(fps):
                    continue
                if fmt not in (rs.format.rgb8, rs.format.bgr8, rs.format.yuyv, rs.format.mjpeg):
                    continue
                candidates.append((w, h, pfps))
            except Exception:
                continue
        if not candidates:
            for p in color_sensor.get_stream_profiles():
                try:
                    if p.stream_type() != rs.stream.color or not p.is_video_stream_profile():
                        continue
                    vp = p.as_video_stream_profile()
                    w, h = int(vp.width()), int(vp.height())
                    pfps = int(vp.fps())
                    fmt = vp.format()
                    if fmt not in (rs.format.rgb8, rs.format.bgr8, rs.format.yuyv, rs.format.mjpeg):
                        continue
                    candidates.append((w, h, pfps))
                except Exception:
                    continue
        if not candidates:
            raise RuntimeError("No usable color stream profiles found.")
        candidates.sort(key=lambda t: (t[0] * t[1], t[2]), reverse=True)
        return candidates[0]

    def start(self) -> None:
        self.output_mp4.parent.mkdir(parents=True, exist_ok=True)
        self._frame_writer.start()

        if self.serial is not None:
            self._config.enable_device(self.serial)

        device = self._select_device(self._rs, self.serial)
        if self.max_color:
            w, h, fps_sel = self._pick_max_color_profile(self._rs, device, int(self.fps))
            self.width, self.height, self.fps = int(w), int(h), int(fps_sel)

        self._config.enable_stream(
            self._rs.stream.color,
            self.width,
            self.height,
            self._rs.format.rgb8,
            self.fps,
        )

        self._pipeline.start(self._config)

        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = self._cv2.VideoWriter(str(self.output_mp4), fourcc, float(self.fps), (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.output_mp4}")

        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)
        try:
            if self._writer is not None:
                self._writer.release()
            if self.show_video:
                self._cv2.destroyAllWindows()
        finally:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._frame_writer.stop()

    def _frame_ts_domain_str(self, domain) -> str:
        # Domain is an enum-like; keep it as string for debugging.
        try:
            return str(domain)
        except Exception:
            return "unknown"

    def _to_sys_time(self, frame_ts_s: float, domain) -> float:
        # frame_ts_s: RealSense get_timestamp() converted from ms to seconds.
        # If librealsense reports SYSTEM_TIME, frame_ts_s is epoch seconds.
        # Otherwise align first frame timestamp to system time.
        if hasattr(self._rs, "timestamp_domain") and domain == self._rs.timestamp_domain.system_time:
            return float(frame_ts_s)

        if self._t_sys0 is None or self._t_cam0_s is None:
            self._t_sys0 = time.time()
            self._t_cam0_s = float(frame_ts_s)

        return float(self._t_sys0) + (float(frame_ts_s) - float(self._t_cam0_s))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            except Exception:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            t_arrival_sys = time.time()
            # librealsense get_timestamp() is in milliseconds; store/export as seconds.
            frame_ts_s = float(color_frame.get_timestamp()) / 1000.0
            domain = color_frame.get_frame_timestamp_domain()
            t_frame_sys = self._to_sys_time(frame_ts_s, domain)

            rgb = self._np.asanyarray(color_frame.get_data())
            bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)

            if self.show_video:
                try:
                    self._cv2.imshow(self.window_name, bgr)
                    key = self._cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._stop.set()
                        continue
                except Exception:
                    # In headless environments imshow may fail; disable preview gracefully.
                    self.show_video = False

            if not self._recording_enabled:
                continue

            self._writer.write(bgr)

            self._frame_writer.put(
                [
                    self._frame_idx,
                    f"{t_frame_sys:.9f}",
                    f"{t_arrival_sys:.9f}",
                    f"{frame_ts_s:.9f}",
                    self._frame_ts_domain_str(domain),
                    self.width,
                    self.height,
                    self.fps,
                ]
            )
            self._frame_idx += 1


class FkImuRecorder:
    def __init__(
        self,
        *,
        session_dir: Path,
        imu_topic: str,
        mocap_topic: str,
        fk_topic: str,
        motor_topic: str,
        yesense_inertial_topic: str,
        save_end_plot: bool = True,
    ):
        self.session_dir = session_dir
        self.imu_topic = imu_topic
        self.mocap_topic = mocap_topic
        self.fk_topic = fk_topic
        self.motor_topic = motor_topic
        self.yesense_inertial_topic = yesense_inertial_topic
        self.save_end_plot = bool(save_end_plot)

        self.rospy, self.Imu, self.PoseStamped, self.MotorPositionsStamped, self.YesenseImuInertialData = (
            _import_ros()
        )

        self._fk_sample_count = 0
        self._motor_sample_count = 0
        self._recording_enabled = False
        self._record_started_sys: Optional[float] = None

        self.imu_writer = CsvWriterThread(
            self.session_dir / "imu.csv",
            header=[
                "t_ros",
                "t_sys_est",
                "qx",
                "qy",
                "qz",
                "qw",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
            ],
        )
        self.fk_writer = CsvWriterThread(
            self.session_dir / "fk.csv",
            header=POSE_CSV_HEADER,
        )
        self.mocap_writer = CsvWriterThread(
            self.session_dir / "mocap.csv",
            header=POSE_CSV_HEADER,
        )
        self.motor_writer = CsvWriterThread(
            self.session_dir / "motor_pos_abs.csv",
            header=MOTOR_POS_CSV_HEADER,
        )
        self.mag_writer = CsvWriterThread(
            self.session_dir / "mag.csv",
            header=[
                "t_ros",
                "t_sys_est",
                "mag_x",
                "mag_y",
                "mag_z",
                "raw_mag_x",
                "raw_mag_y",
                "raw_mag_z",
            ],
        )

        self._clock_sync: Optional[RosSysClockSync] = None

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.imu_writer.start()
        self.fk_writer.start()
        self.mocap_writer.start()
        self.motor_writer.start()
        self.mag_writer.start()

        self.rospy.init_node("record_fk_imu_video", anonymous=False)
        self._clock_sync = RosSysClockSync(
            ros0=float(self.rospy.Time.now().to_sec()),
            sys0=float(time.time()),
        )

        print(f"[INFO] FK uses subscribed pose topic {self.fk_topic}", file=sys.stderr)

        self.rospy.Subscriber(self.imu_topic, self.Imu, self._imu_cb, queue_size=POSE_SUBSCRIBER_QUEUE_SIZE)
        self.rospy.Subscriber(
            self.mocap_topic,
            self.PoseStamped,
            self._mocap_cb,
            queue_size=POSE_SUBSCRIBER_QUEUE_SIZE,
        )
        self.rospy.Subscriber(
            self.fk_topic,
            self.PoseStamped,
            self._fk_pose_cb,
            queue_size=POSE_SUBSCRIBER_QUEUE_SIZE,
        )
        self.rospy.Subscriber(
            self.motor_topic,
            self.MotorPositionsStamped,
            self._motor_pos_cb,
            queue_size=POSE_SUBSCRIBER_QUEUE_SIZE,
        )
        self.rospy.Subscriber(
            self.yesense_inertial_topic,
            self.YesenseImuInertialData,
            self._yesense_mag_cb,
            queue_size=POSE_SUBSCRIBER_QUEUE_SIZE,
        )
        print(f"[INFO] FK subscribes {self.fk_topic}", file=sys.stderr)
        print(f"[INFO] Motor subscribes {self.motor_topic}", file=sys.stderr)
        print(f"[INFO] Magnetometer subscribes {self.yesense_inertial_topic}", file=sys.stderr)
        try:
            self.rospy.wait_for_message(self.fk_topic, self.PoseStamped, timeout=20.0)
            print(f"[INFO] First FK pose message received from {self.fk_topic}.", file=sys.stderr)
        except self.rospy.ROSException as e:
            raise RuntimeError(
                f"No message on {self.fk_topic} within 20s. "
                "Please start the EKF node that publishes fk_pose first."
            ) from e

    def begin_recording(self) -> None:
        """Enable writing IMU/mocap/FK samples to disk (after warmup)."""
        self._recording_enabled = True
        self._record_started_sys = time.time()

    def stop(self) -> None:
        self.imu_writer.stop()
        self.fk_writer.stop()
        self.mocap_writer.stop()
        self.motor_writer.stop()
        self.mag_writer.stop()
        if self._fk_sample_count == 0:
            print(
                f"[WARN] fk.csv has no samples. Ensure {self.fk_topic} is publishing.",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] FK samples recorded: {self._fk_sample_count}",
                file=sys.stderr,
            )
        print(
            f"[INFO] motor_pos_abs samples recorded: {self._motor_sample_count}",
            file=sys.stderr,
        )

    def save_plot_png(self, path: Path) -> bool:
        if not self.save_end_plot:
            return False
        out = plot_fk_mocap_from_csv(self.session_dir, output_png=path)
        return out is not None

    def _imu_cb(self, msg) -> None:
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return

        o = msg.orientation
        la = msg.linear_acceleration
        av = msg.angular_velocity
        self.imu_writer.put(
            [
                f"{t_ros:.9f}",
                f"{t_sys_est:.9f}",
                f"{o.x:.10g}",
                f"{o.y:.10g}",
                f"{o.z:.10g}",
                f"{o.w:.10g}",
                f"{la.x:.10g}",
                f"{la.y:.10g}",
                f"{la.z:.10g}",
                f"{av.x:.10g}",
                f"{av.y:.10g}",
                f"{av.z:.10g}",
            ]
        )

    def _mocap_cb(self, msg) -> None:
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return
        self.mocap_writer.put(_pose_msg_to_csv_row(msg, t_ros, t_sys_est))

    def _fk_pose_cb(self, msg) -> None:
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return

        self.fk_writer.put(_pose_msg_to_csv_row(msg, t_ros, t_sys_est))
        self._fk_sample_count += 1

    def _motor_pos_cb(self, msg) -> None:
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return

        self.motor_writer.put(_motor_pos_msg_to_csv_row(msg, t_ros, t_sys_est))
        self._motor_sample_count += 1

    def _yesense_mag_cb(self, msg) -> None:
        if not self._recording_enabled:
            return
        assert self._clock_sync is not None
        t_ros = float(self.rospy.Time.now().to_sec())
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        m = msg.magnetic
        rm = msg.raw_magnetic
        self.mag_writer.put(
            [
                f"{t_ros:.9f}",
                f"{t_sys_est:.9f}",
                f"{m.x:.10g}",
                f"{m.y:.10g}",
                f"{m.z:.10g}",
                f"{rm.x:.10g}",
                f"{rm.y:.10g}",
                f"{rm.z:.10g}",
            ]
        )

    def metadata(self) -> dict:
        assert self._clock_sync is not None
        return {
            "imu_topic": self.imu_topic,
            "mocap_topic": self.mocap_topic,
            "fk_topic": self.fk_topic,
            "motor_topic": self.motor_topic,
            "yesense_inertial_topic": self.yesense_inertial_topic,
            "fk_samples": int(self._fk_sample_count),
            "motor_samples": int(self._motor_sample_count),
            "record_started_sys": self._record_started_sys,
            "clock_sync": {
                "ros0": float(self._clock_sync.ros0),
                "sys0": float(self._clock_sync.sys0),
                "note": "t_sys_est = sys0 + (t_ros - ros0); valid only if ROS wall-clock time is used.",
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record IMU + FK(pose topic) + D455 video with aligned timestamps.")

    # Session/output
    p.add_argument("--output-root", type=Path, default=Path("recordings"), help="Root output folder.")
    p.add_argument("--session-name", type=str, default="", help="Optional session folder name override.")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to record after warmup. 0 = until Ctrl+C.")
    p.add_argument(
        "--warmup-sec",
        type=float,
        default=2.0,
        help="Warmup time after startup (video/ROS/FK run but no data written).",
    )
    p.add_argument(
        "--video-only",
        action="store_true",
        help="Record ONLY D455 video + video_frames.csv (no ROS, no IMU/FK).",
    )

    # ROS
    p.add_argument("--imu-topic", type=str, default="/imu", help="IMU topic (sensor_msgs/Imu).")
    p.add_argument("--mocap-topic", type=str, default="/vrpn_client_node/cdpr/pose", help="Mocap pose topic (PoseStamped).")
    p.add_argument("--fk-topic", type=str, default="/fk_pose", help="FK pose topic (PoseStamped, from ekf node).")
    p.add_argument(
        "--motor-topic",
        type=str,
        default="/motor_pos_abs",
        help="Motor absolute positions topic (MotorPositionsStamped).",
    )
    p.add_argument(
        "--yesense-inertial-topic",
        type=str,
        default="/yesense/inertial_data",
        help="Yesense inertial topic; only magnetometer fields are saved to mag.csv.",
    )

    # Camera
    p.add_argument("--serial", type=str, default=None, help="RealSense serial.")
    p.add_argument("--fps", type=int, default=30, help="Video FPS.")
    p.add_argument("--width", type=int, default=1280, help="Video width (ignored if --max-color).")
    p.add_argument("--height", type=int, default=720, help="Video height (ignored if --max-color).")
    p.add_argument("--max-color", action="store_true", help="Auto-pick max supported color resolution (at --fps when possible).")
    p.add_argument(
        "--no-show-video",
        dest="show_video",
        action="store_false",
        help="Disable live camera preview window.",
    )
    p.set_defaults(show_video=True)

    # Plot (post-recording only)
    p.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Skip saving fk_mocap_compare.png after recording.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    session_name = args.session_name.strip() or f"session_{_timestamp_tag()}"
    session_dir = (args.output_root / session_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    video_mp4 = session_dir / "video.mp4"
    video_frames_csv = session_dir / "video_frames.csv"
    metadata_path = session_dir / "metadata.json"

    stop_flag = threading.Event()

    def _handle_sigint(_sig, _frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    fkimu = None
    if not args.video_only:
        fkimu = FkImuRecorder(
            session_dir=session_dir,
            imu_topic=args.imu_topic,
            mocap_topic=args.mocap_topic,
            fk_topic=args.fk_topic,
            motor_topic=args.motor_topic,
            yesense_inertial_topic=args.yesense_inertial_topic,
            save_end_plot=not bool(args.no_save_plot),
        )
        fkimu.start()

    if args.video_only and not ENABLE_CAMERA_RECORDING:
        raise RuntimeError(
            "ENABLE_CAMERA_RECORDING is False but --video-only was set. "
            "Enable camera recording or remove --video-only."
        )

    cam: Optional[D455VideoRecorder] = None
    if ENABLE_CAMERA_RECORDING:
        # Start camera recorder.
        cam = D455VideoRecorder(
            serial=args.serial,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            max_color=bool(args.max_color),
            show_video=bool(args.show_video),
            output_mp4=video_mp4,
            output_frames_csv=video_frames_csv,
        )
        cam.start()
    else:
        print("[INFO] Camera recording disabled by ENABLE_CAMERA_RECORDING.", file=sys.stderr)

    # Write metadata (initial).
    meta: dict[str, Any] = {
        "session_dir": str(session_dir),
        "created_at": _dt.datetime.now().isoformat(),
        "video": (
            {
                "enabled": True,
                "serial": args.serial,
                "fps": cam.fps if cam is not None else int(args.fps),
                "width": cam.width if cam is not None else int(args.width),
                "height": cam.height if cam is not None else int(args.height),
                "max_color": bool(args.max_color),
                "mp4": str(video_mp4),
                "frames_csv": str(video_frames_csv),
                "alignment": {
                    "time_unit": "s",
                    "frame_timestamp_domain": "SYSTEM_TIME: frame_ts is epoch s; otherwise first-frame alignment to t_frame_sys",
                },
            }
            if ENABLE_CAMERA_RECORDING
            else {
                "enabled": False,
                "reason": "disabled_by_ENABLE_CAMERA_RECORDING",
            }
        ),
        "ros": (fkimu.metadata() if fkimu is not None else None),
    }
    plot_path = session_dir / "fk_mocap_compare.png"

    print(f"[INFO] session_dir={session_dir}")
    if ENABLE_CAMERA_RECORDING:
        print(f"[INFO] video={video_mp4}  frames={video_frames_csv}")
    else:
        print("[INFO] video recording is disabled.")
    print(
        f"[INFO] imu={session_dir/'imu.csv'}  mag={session_dir/'mag.csv'}  "
        f"fk={session_dir/'fk.csv'}  mocap={session_dir/'mocap.csv'}  "
        f"motor={session_dir/'motor_pos_abs.csv'}"
    )
    if fkimu is not None and not args.no_save_plot:
        print(f"[INFO] post-recording plot -> {plot_path}")

    warmup_sec = max(0.0, float(args.warmup_sec))
    if warmup_sec > 0:
        print(f"[INFO] Warming up {warmup_sec:.1f}s (camera/ROS/FK active, not writing files)...")
        t_warmup_end = time.time() + warmup_sec
        while not stop_flag.is_set() and time.time() < t_warmup_end:
            time.sleep(0.02)

    if fkimu is not None:
        fkimu.begin_recording()
    if cam is not None:
        cam.begin_recording()
    print("[INFO] Recording started.")

    meta["record_warmup_sec"] = warmup_sec
    meta["recording_started_at"] = _dt.datetime.now().isoformat()
    if fkimu is not None:
        meta["ros"] = fkimu.metadata()
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] duration={'until Ctrl+C' if args.duration <= 0 else f'{args.duration}s (after warmup)'}")
    t0 = time.time()
    try:
        while not stop_flag.is_set():
            if args.duration and args.duration > 0 and (time.time() - t0) >= float(args.duration):
                break
            time.sleep(0.02)
    finally:
        if cam is not None:
            cam.stop()
        if fkimu is not None:
            fkimu.stop()
            compare_fk_motor_stamp_duplicates(session_dir)
            # Save comparison plot from CSV.
            try:
                if fkimu.save_plot_png(plot_path):
                    print(f"[INFO] Saved FK vs mocap plot: {plot_path}")
            except Exception as e:
                print(f"[WARN] Failed to save plot: {e}", file=sys.stderr)

    print("[INFO] done.")


if __name__ == "__main__":
    main()

