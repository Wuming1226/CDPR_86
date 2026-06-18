#!/usr/bin/env python3
"""
Record IMU + FK pose topic + RealSense D455 color video with aligned timestamps.

Outputs a session folder:
  - video.mp4
  - video_frames.csv
  - imu.csv
  - fk.csv
  - mocap.csv
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
import os
import queue
from collections import deque
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RECORDINGS_ROOT = _SCRIPT_DIR.parent / "recordings"


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
        from cdpr_86_msgs.msg import CableLengthsStamped  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ROS python environment is not available.\n"
            "Fix by sourcing your ROS + workspace setup, e.g.:\n"
            "  source /opt/ros/<distro>/setup.bash\n"
            "  source <your_catkin_ws>/devel/setup.bash\n"
            "Then run this script again.\n"
            "If you only want to record video (no ROS), add: --video-only"
        ) from e

    return rospy, Imu, PoseStamped, CableLengthsStamped


def _import_fk():
    from cdpr_euler_ekf import (
        CDPRGeometry,
        cable_lengths_from_pose,
        cdpr_geometry_from_calibration_file,
        forward_kinematics_lm,
        forward_kinematics_lm_with_prior,
        make_demo_geometry,
    )

    return (
        CDPRGeometry,
        cable_lengths_from_pose,
        cdpr_geometry_from_calibration_file,
        forward_kinematics_lm,
        forward_kinematics_lm_with_prior,
        make_demo_geometry,
    )


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


class RollingPoseBuffer:
    """Rolling window buffer for 6D pose [x,y,z,roll,pitch,yaw] vs time."""

    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.t: deque = deque()
        self.values = [deque() for _ in range(6)]

    def append(self, stamp_sec: float, vec6) -> None:
        self.t.append(float(stamp_sec))
        for i in range(6):
            self.values[i].append(float(vec6[i]))
        self.trim(float(stamp_sec))

    def trim(self, newest_sec: float) -> None:
        threshold = newest_sec - self.window_sec
        while self.t and self.t[0] < threshold:
            self.t.popleft()
            for i in range(6):
                self.values[i].popleft()


def _pose_msg_to_xyzrpy(msg) -> list[float]:
    """PoseStamped -> [x,y,z,roll,pitch,yaw] (rad), SciPy ZYX convention."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    p = msg.pose.position
    q = msg.pose.orientation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        return [float(p.x), float(p.y), float(p.z), 0.0, 0.0, 0.0]
    quat = quat / np.linalg.norm(quat)
    yaw, pitch, roll = R.from_quat(quat).as_euler("ZYX", degrees=False)
    return [float(p.x), float(p.y), float(p.z), float(roll), float(pitch), float(yaw)]


class FkMocapLivePlot:
    """Live matplotlib plot: mocap vs FK during recording."""

    def __init__(self, window_sec: float = 30.0):
        import matplotlib.pyplot as plt

        self.window_sec = float(window_sec)
        self._plt = plt
        self.fig, self.axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        self.fig.suptitle("FK vs Mocap (rolling window)")
        self.labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
        self.lines_mocap = []
        self.lines_fk = []
        for i, ax in enumerate(self.axes):
            line_m, = ax.plot([], [], "b-", linewidth=1.4, label="mocap")
            line_f, = ax.plot([], [], "m-", linewidth=1.2, label="fk")
            self.lines_mocap.append(line_m)
            self.lines_fk.append(line_f)
            ax.set_ylabel(self.labels[i])
            ax.grid(True, alpha=0.3)
        self.axes[-1].set_xlabel("time [s], relative to now")
        self.axes[0].legend(loc="upper left")
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.ion()
        plt.show(block=False)

    def update(self, mocap_buf: RollingPoseBuffer, fk_buf: RollingPoseBuffer, now_sec: float) -> None:
        import numpy as np

        mocap_buf.trim(now_sec)
        fk_buf.trim(now_sec)

        t_m = np.array(mocap_buf.t, dtype=float)
        t_f = np.array(fk_buf.t, dtype=float)
        x_m = t_m - now_sec if t_m.size else np.array([])
        x_f = t_f - now_sec if t_f.size else np.array([])

        for i in range(6):
            y_m = np.array(mocap_buf.values[i], dtype=float)
            y_f = np.array(fk_buf.values[i], dtype=float)
            if i >= 3:
                y_m = np.rad2deg(y_m)
                y_f = np.rad2deg(y_f)
            self.lines_mocap[i].set_data(x_m, y_m)
            self.lines_fk[i].set_data(x_f, y_f)
            ax = self.axes[i]
            ax.set_xlim(-self.window_sec, 0.0)
            y_all = np.hstack([y_m, y_f]) if (y_m.size + y_f.size) > 0 else np.array([0.0])
            y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
            center = 0.5 * (y_min + y_max)
            min_span = 0.02 if i < 3 else 2.0
            half = 0.5 * max(y_max - y_min, min_span)
            ax.set_ylim(center - half, center + half)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_png(self, path: Path) -> None:
        self.fig.savefig(path, dpi=150)

    def close(self) -> None:
        self._plt.close(self.fig)


def plot_fk_mocap_from_csv(session_dir: Path, output_png: Optional[Path] = None) -> Optional[Path]:
    """Plot FK vs mocap from saved CSV files (post-recording)."""
    fk_path = session_dir / "fk.csv"
    mocap_path = session_dir / "mocap.csv"
    if not fk_path.is_file() or not mocap_path.is_file():
        return None

    import numpy as np
    import matplotlib.pyplot as plt

    def _load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["t_sys_est"])
                pose = [float(row[c]) for c in ("x", "y", "z", "roll", "pitch", "yaw")]
                rows.append((t, pose))
        if not rows:
            return np.array([]), np.empty((0, 6))
        t_arr = np.array([r[0] for r in rows], dtype=float)
        pose_arr = np.array([r[1] for r in rows], dtype=float)
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
        calibration_file: str,
        use_calibrated_geometry: bool,
        use_calibrated_cable_length: bool,
        fk_use_prior: bool,
        fk_max_iters: int,
        fk_prior_pos_weight: float,
        fk_prior_att_weight: float,
        fk_seed_rho: Optional[list[float]],
        enable_plot: bool = True,
        plot_window_sec: float = 30.0,
    ):
        self.session_dir = session_dir
        self.imu_topic = imu_topic
        self.mocap_topic = mocap_topic
        self.fk_topic = fk_topic
        self.calibration_file = calibration_file
        self.use_calibrated_geometry = bool(use_calibrated_geometry)
        self.use_calibrated_cable_length = bool(use_calibrated_cable_length)
        self.fk_use_prior = bool(fk_use_prior)
        self.fk_max_iters = int(fk_max_iters)
        self.fk_prior_pos_weight = float(fk_prior_pos_weight)
        self.fk_prior_att_weight = float(fk_prior_att_weight)
        self.enable_plot = bool(enable_plot)
        self.plot_window_sec = float(plot_window_sec)

        self.rospy, self.Imu, self.PoseStamped, _CableLengthsStamped = _import_ros()
        (
            self._CDPRGeometry,
            _cable_lengths_from_pose,
            cdpr_geometry_from_calibration_file,
            forward_kinematics_lm,
            forward_kinematics_lm_with_prior,
            make_demo_geometry,
        ) = _import_fk()
        self._cdpr_geometry_from_calibration_file = cdpr_geometry_from_calibration_file
        self._make_demo_geometry = make_demo_geometry

        self.cdpr = None  # kept for backward-compatible metadata shape
        if self.use_calibrated_geometry:
            self.geom = self._cdpr_geometry_from_calibration_file(
                self.calibration_file,
                base_dir=Path(__file__).resolve().parent,
            )
        else:
            self.geom = self._make_demo_geometry(use_ros_cdpr=False)

        # FK seed [x, y, z, roll, pitch, yaw] in radians.
        if fk_seed_rho is not None:
            if len(fk_seed_rho) != 6:
                raise ValueError("--fk-seed-rho must have 6 floats: x y z roll pitch yaw (rad)")
            self.rho_fk_seed = self._np_array(fk_seed_rho)
        else:
            # Reasonable default close to typical workspace (matches demo in cdpr_euler_ekf.py).
            self.rho_fk_seed = self._np_array([-1.25, -0.74, 1.55, 0.0, 0.0, 0.0])

        self._prior_weights = self._np_array(
            [
                self.fk_prior_pos_weight,
                self.fk_prior_pos_weight,
                self.fk_prior_pos_weight,
                self.fk_prior_att_weight,
                self.fk_prior_att_weight,
                self.fk_prior_att_weight,
            ]
        )

        self._lock = threading.Lock()
        self._latest_imu = None  # last imu msg (for metadata/debug)
        self._fk_sample_count = 0
        self._cable_topic_count = 0
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
            header=[
                "t_ros",
                "t_sys_est",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        )
        self.mocap_writer = CsvWriterThread(
            self.session_dir / "mocap.csv",
            header=[
                "t_ros",
                "t_sys_est",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        )

        self._clock_sync: Optional[RosSysClockSync] = None
        self._plot_lock = threading.Lock()
        self._mocap_buf = RollingPoseBuffer(self.plot_window_sec)
        self._fk_buf = RollingPoseBuffer(self.plot_window_sec)
        self._live_plot: Optional[FkMocapLivePlot] = None
        if self.enable_plot:
            try:
                self._live_plot = FkMocapLivePlot(window_sec=self.plot_window_sec)
            except Exception as e:
                print(f"[WARN] Live plot disabled: {e}", file=sys.stderr)
                self._live_plot = None

    @staticmethod
    def _np_array(x):
        import numpy as np  # local import

        return np.asarray(x, dtype=float).reshape(-1)

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.imu_writer.start()
        self.fk_writer.start()
        self.mocap_writer.start()

        self.rospy.init_node("record_fk_imu_video", anonymous=False)
        self._clock_sync = RosSysClockSync(
            ros0=float(self.rospy.Time.now().to_sec()),
            sys0=float(time.time()),
        )

        print(f"[INFO] FK uses subscribed pose topic {self.fk_topic}", file=sys.stderr)

        self.rospy.Subscriber(self.imu_topic, self.Imu, self._imu_cb, queue_size=1000)
        self.rospy.Subscriber(self.mocap_topic, self.PoseStamped, self._mocap_cb, queue_size=200)
        self.rospy.Subscriber(self.fk_topic, self.PoseStamped, self._fk_pose_cb, queue_size=1000)
        print(f"[INFO] FK subscribes {self.fk_topic}", file=sys.stderr)
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
        with self._plot_lock:
            self._mocap_buf = RollingPoseBuffer(self.plot_window_sec)
            self._fk_buf = RollingPoseBuffer(self.plot_window_sec)

    def stop(self) -> None:
        self.imu_writer.stop()
        self.fk_writer.stop()
        self.mocap_writer.stop()
        if self._fk_sample_count == 0:
            print(
                f"[WARN] fk.csv has no samples. Ensure {self.fk_topic} is publishing.",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] FK samples recorded: {self._fk_sample_count} (fk_pose_cb={self._cable_topic_count})",
                file=sys.stderr,
            )
        if self._live_plot is not None:
            try:
                self._live_plot.close()
            except Exception:
                pass
            self._live_plot = None

    def update_plot(self) -> None:
        if self._live_plot is None:
            return
        now = float(self.rospy.Time.now().to_sec())
        with self._plot_lock:
            self._live_plot.update(self._mocap_buf, self._fk_buf, now)

    def save_plot_png(self, path: Path) -> bool:
        if self._live_plot is not None:
            try:
                self._live_plot.save_png(path)
                return True
            except Exception:
                pass
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
        with self._lock:
            self._latest_imu = msg

    def _mocap_cb(self, msg) -> None:
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return
        xyzrpy = _pose_msg_to_xyzrpy(msg)
        self.mocap_writer.put(
            [f"{t_ros:.9f}", f"{t_sys_est:.9f}"]
            + [f"{v:.10g}" for v in xyzrpy]
        )
        with self._plot_lock:
            self._mocap_buf.append(t_ros, xyzrpy)

    def _fk_pose_cb(self, msg) -> None:
        self._cable_topic_count += 1
        t_ros = _ros_time_to_sec(msg.header.stamp, self.rospy)
        assert self._clock_sync is not None
        t_sys_est = self._clock_sync.ros_to_sys(t_ros)
        if not self._recording_enabled:
            return

        x, y, z, roll, pitch, yaw = _pose_msg_to_xyzrpy(msg)

        row = [
            f"{t_ros:.9f}",
            f"{t_sys_est:.9f}",
            f"{x:.10g}",
            f"{y:.10g}",
            f"{z:.10g}",
            f"{roll:.10g}",
            f"{pitch:.10g}",
            f"{yaw:.10g}",
        ]

        self.fk_writer.put(row)
        self._fk_sample_count += 1
        with self._plot_lock:
            self._fk_buf.append(t_ros, [x, y, z, roll, pitch, yaw])

    def metadata(self) -> dict:
        assert self._clock_sync is not None
        return {
            "imu_topic": self.imu_topic,
            "mocap_topic": self.mocap_topic,
            "fk_topic": self.fk_topic,
            "fk_samples": int(self._fk_sample_count),
            "record_started_sys": self._record_started_sys,
            "cdpr": {
                "is_calibrated": self.use_calibrated_geometry,
                "use_calibrated_cable_length": self.use_calibrated_cable_length,
                "calibration_file": str(self.calibration_file) if self.use_calibrated_geometry else None,
                "init_cable_lens_l0": None,
                "init_motor_pos_abs": None,
            },
            "geometry": {
                "use_calibrated_geometry": self.use_calibrated_geometry,
                "calibration_file": self.calibration_file if self.use_calibrated_geometry else None,
                "m": int(self.geom.m) if self.geom is not None else None,
            },
            "fk": {
                "fk_use_prior": self.fk_use_prior,
                "fk_max_iters": self.fk_max_iters,
                "fk_prior_pos_weight": self.fk_prior_pos_weight,
                "fk_prior_att_weight": self.fk_prior_att_weight,
                "fk_seed_rho": [float(v) for v in self.rho_fk_seed.tolist()],
            },
            "clock_sync": {
                "ros0": float(self._clock_sync.ros0),
                "sys0": float(self._clock_sync.sys0),
                "note": "t_sys_est = sys0 + (t_ros - ros0); valid only if ROS wall-clock time is used.",
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record IMU + FK(pose topic) + D455 video with aligned timestamps.")

    # Session/output
    p.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_RECORDINGS_ROOT,
        help=f"Root output folder (default: {_DEFAULT_RECORDINGS_ROOT}).",
    )
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

    # FK
    p.add_argument(
        "--no-calibrated-geometry",
        dest="use_calibrated_geometry",
        action="store_false",
        help="Use nominal geometry (debug only). Default uses calibrated geometry.",
    )
    p.set_defaults(use_calibrated_geometry=True)
    p.add_argument("--calibration-file", type=str, default="cdpr_kinematic_calib.json", help="Calibration JSON path (a,b keys).")
    p.add_argument(
        "--use-calibrated-cable-length",
        dest="use_calibrated_cable_length",
        action="store_true",
        help="Use l0/init_motor_pos_abs from calibration json. Default matches EKF launch: mocap init l0.",
    )
    p.set_defaults(use_calibrated_cable_length=False)
    p.add_argument("--fk-no-prior", action="store_true", help="Disable FK prior term (less stable).")
    p.add_argument("--fk-max-iters", type=int, default=20, help="FK LM max iterations per cable sample.")
    p.add_argument("--fk-prior-pos-weight", type=float, default=.0, help="FK prior weight on xyz.")
    p.add_argument("--fk-prior-att-weight", type=float, default=.0, help="FK prior weight on rpy continuity.")
    p.add_argument(
        "--fk-seed-rho",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Initial FK seed rho: x y z roll pitch yaw (radians).",
    )

    # Camera
    p.add_argument("--serial", type=str, default=None, help="RealSense serial.")
    p.add_argument("--fps", type=int, default=30, help="Video FPS.")
    p.add_argument("--width", type=int, default=640, help="Video width (ignored if --max-color).")
    p.add_argument("--height", type=int, default=480, help="Video height (ignored if --max-color).")
    p.add_argument("--max-color", action="store_true", help="Auto-pick max supported color resolution (at --fps when possible).")
    p.add_argument(
        "--no-show-video",
        dest="show_video",
        action="store_false",
        help="Disable live camera preview window.",
    )
    p.set_defaults(show_video=True)

    # Plot
    p.add_argument("--no-plot", action="store_true", help="Disable live FK vs mocap plot during recording.")
    p.add_argument("--plot-window", type=float, default=30.0, help="Rolling plot window length in seconds.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    session_name = args.session_name.strip() or f"session_{_timestamp_tag()}"
    output_root = args.output_root.resolve()
    session_dir = (output_root / session_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] session_dir={session_dir}")

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
        # Start ROS recorder (subscribers + FK solve).
        fkimu = FkImuRecorder(
            session_dir=session_dir,
            imu_topic=args.imu_topic,
            mocap_topic=args.mocap_topic,
            fk_topic=args.fk_topic,
            calibration_file=args.calibration_file,
            use_calibrated_geometry=bool(args.use_calibrated_geometry),
            use_calibrated_cable_length=bool(args.use_calibrated_cable_length),
            fk_use_prior=not bool(args.fk_no_prior),
            fk_max_iters=int(args.fk_max_iters),
            fk_prior_pos_weight=float(args.fk_prior_pos_weight),
            fk_prior_att_weight=float(args.fk_prior_att_weight),
            fk_seed_rho=args.fk_seed_rho,
            enable_plot=not bool(args.no_plot),
            plot_window_sec=float(args.plot_window),
        )
        fkimu.start()

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

    # Write metadata (initial).
    meta: dict[str, Any] = {
        "session_dir": str(session_dir),
        "created_at": _dt.datetime.now().isoformat(),
        "video": {
            "serial": args.serial,
            "fps": cam.fps,
            "width": cam.width,
            "height": cam.height,
            "max_color": bool(args.max_color),
            "mp4": str(video_mp4),
            "frames_csv": str(video_frames_csv),
            "alignment": {
                "time_unit": "s",
                "frame_timestamp_domain": "SYSTEM_TIME: frame_ts is epoch s; otherwise first-frame alignment to t_frame_sys",
            },
        },
        "ros": (fkimu.metadata() if fkimu is not None else None),
    }
    plot_path = session_dir / "fk_mocap_compare.png"

    print(f"[INFO] video={video_mp4}  frames={video_frames_csv}")
    print(f"[INFO] imu={session_dir/'imu.csv'}  fk={session_dir/'fk.csv'}  mocap={session_dir/'mocap.csv'}")
    if fkimu is not None and not args.no_plot:
        print(f"[INFO] live plot: FK vs mocap (window={args.plot_window}s); save -> {plot_path}")

    warmup_sec = max(0.0, float(args.warmup_sec))
    if warmup_sec > 0:
        print(f"[INFO] Warming up {warmup_sec:.1f}s (camera/ROS/FK active, not writing files)...")
        t_warmup_end = time.time() + warmup_sec
        while not stop_flag.is_set() and time.time() < t_warmup_end:
            time.sleep(0.02)

    if fkimu is not None:
        fkimu.begin_recording()
    cam.begin_recording()
    print("[INFO] Recording started.")

    meta["record_warmup_sec"] = warmup_sec
    meta["recording_started_at"] = _dt.datetime.now().isoformat()
    if fkimu is not None:
        meta["ros"] = fkimu.metadata()
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] duration={'until Ctrl+C' if args.duration <= 0 else f'{args.duration}s (after warmup)'}")
    t0 = time.time()
    last_plot = 0.0
    plot_interval = 0.1
    try:
        while not stop_flag.is_set():
            if args.duration and args.duration > 0 and (time.time() - t0) >= float(args.duration):
                break
            now_loop = time.time()
            if fkimu is not None and fkimu._live_plot is not None and (now_loop - last_plot) >= plot_interval:
                try:
                    fkimu.update_plot()
                except Exception:
                    pass
                last_plot = now_loop
            time.sleep(0.02)
    finally:
        cam.stop()
        if fkimu is not None:
            fkimu.stop()
            # Save comparison plot from live window or CSV.
            try:
                if fkimu.save_plot_png(plot_path):
                    print(f"[INFO] Saved FK vs mocap plot: {plot_path}")
            except Exception as e:
                print(f"[WARN] Failed to save plot: {e}", file=sys.stderr)

    print("[INFO] done.")


if __name__ == "__main__":
    main()

