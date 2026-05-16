#!/usr/bin/env python3
import math
from collections import deque
from typing import Deque, Optional, Tuple

import message_filters
import numpy as np
import rospy
from cdpr_86_host.msg import CableLengthsStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R

from cdpr import CDPR
from imu_extrinsic import ImuExtrinsic, load_extrinsic_for_node
from cdpr_euler_ekf import (
    CDPRGeometry,
    EulerEKFCDPR,
    forward_kinematics_lm,
    forward_kinematics_lm_with_prior,
    forward_kinematics_lm_xyz_with_fixed_attitude,
    forward_kinematics_lm_xyz_with_fixed_attitude_and_prior,
)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _quat_valid(q: np.ndarray) -> bool:
    q = np.asarray(q, dtype=float).reshape(4)
    return bool(np.linalg.norm(q) > 1e-9 and np.isfinite(q).all())


class RollingRateTracker:
    """Rolling-window EKF callback rate from message timestamps (seconds)."""

    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self._t: Deque[float] = deque()

    def record(self, stamp_sec: float) -> None:
        self._t.append(stamp_sec)
        threshold = stamp_sec - self.window_sec
        while self._t and self._t[0] < threshold:
            self._t.popleft()

    def snapshot(self) -> Optional[dict]:
        if len(self._t) < 2:
            return None
        t_arr = np.asarray(self._t, dtype=float)
        span = float(t_arr[-1] - t_arr[0])
        if span <= 1e-9:
            return None
        dt_arr = np.diff(t_arr)
        dt_arr = dt_arr[(dt_arr > 1e-9) & np.isfinite(dt_arr)]
        out = {
            "n": len(t_arr),
            "span_s": span,
            "rate_hz": (len(t_arr) - 1) / span,
            "inst_hz": None,
            "dt_mean": None,
            "dt_std": None,
            "dt_min": None,
            "dt_max": None,
        }
        if dt_arr.size:
            out["inst_hz"] = 1.0 / float(dt_arr[-1])
            out["dt_mean"] = float(dt_arr.mean())
            out["dt_std"] = float(dt_arr.std(ddof=1)) if dt_arr.size > 1 else 0.0
            out["dt_min"] = float(dt_arr.min())
            out["dt_max"] = float(dt_arr.max())
        return out


class CDPREulerEkfNode:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.pose_topic = rospy.get_param("~pose_topic", "/ekf_pose")
        self.fk_pose_topic = rospy.get_param("~fk_pose_topic", "/fk_pose")
        self.mocap_topic = rospy.get_param("~mocap_topic", "/vrpn_client_node/cdpr/pose")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.cable_topic = rospy.get_param("~cable_topic", "/cable_lengths_measure")
        self.default_dt = float(rospy.get_param("~default_dt", 0.01))
        self.fk_max_iters = int(rospy.get_param("~fk_max_iters", 20))
        self.fk_use_prior = _as_bool(rospy.get_param("~fk_use_prior", True))
        self.fk_prior_pos_weight = float(rospy.get_param("~fk_prior_pos_weight", 2.0))
        self.fk_prior_att_weight = float(rospy.get_param("~fk_prior_att_weight", 10.0))
        self.fk_with_given_rpy = _as_bool(rospy.get_param("~fk_with_given_rpy", False))
        self.rpy_from_imu = _as_bool(rospy.get_param("~rpy_from_imu", False))
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 100))
        self.sync_slop = float(rospy.get_param("~sync_slop", 1.0 / 15))
        self.is_calibrated = _as_bool(rospy.get_param("~is_calibrated", True))
        self.use_calibrated_cable_length = _as_bool(rospy.get_param("~use_calibrated_cable_length", True))
        self.calibration_file = rospy.get_param("~calibration_file", "cdpr_kinematic_calib.json")
        self.mocap_init_timeout = float(rospy.get_param("~mocap_init_timeout", 5.0))
        self.log_ekf_rate = _as_bool(rospy.get_param("~log_ekf_rate", True))
        self.rate_window_sec = float(rospy.get_param("~rate_window_sec", 10.0))
        self.rate_log_period = float(rospy.get_param("~rate_log_period", 2.0))
        self.apply_imu_extrinsic = _as_bool(rospy.get_param("~apply_imu_extrinsic", True))
        self.imu_extrinsic_file = rospy.get_param("~imu_extrinsic_file", "cdpr_imu_extrinsic.json")
        self._imu_extrinsic: Optional[ImuExtrinsic] = None
        if self.rpy_from_imu and self.apply_imu_extrinsic:
            self._imu_extrinsic = load_extrinsic_for_node(
                self.imu_extrinsic_file,
                enabled=True,
                node_name="cdpr_euler_ekf_node",
            )

        g_a = np.array([0.0, 0.0, -9.81], dtype=float)
        self.cdpr = CDPR(
            imu_active=self.rpy_from_imu,
            is_calibrated=self.is_calibrated,
            use_calibrated_cable_length=self.use_calibrated_cable_length,
            calibration_file=(self.calibration_file if self.is_calibrated else None),
            imu_extrinsic=self._imu_extrinsic,
            apply_imu_extrinsic=self.apply_imu_extrinsic,
            imu_extrinsic_file=self.imu_extrinsic_file,
        )
        wa, wb = self.cdpr.get_cable_attachment_points()
        self.geom = CDPRGeometry(winches_a=wa, attachments_b=wb)
        self.ekf = EulerEKFCDPR(dt=self.default_dt, g_a=g_a)

        x0 = np.zeros(15, dtype=float)
        x, y, z, quat = self._wait_valid_mocap_init_pose()
        quat_arr = np.asarray(quat, dtype=float).reshape(4)
        quat_arr = quat_arr / np.linalg.norm(quat_arr)
        yaw, pitch, roll = R.from_quat(quat_arr).as_euler("ZYX", degrees=False)
        x0[0:3] = np.array([x, y, z], dtype=float)
        x0[6:9] = np.array([roll, pitch, yaw], dtype=float)
        p0 = np.eye(15, dtype=float) * 0.03
        p0[6:9, 6:9] *= 4.0
        self.ekf.set_initial(x0, p0)

        self.rho_fk_seed = np.hstack([self.ekf.x[0:3], self.ekf.x[6:9]])
        self.r_fk_seed = self.ekf.x[0:3].copy()
        print(self.rho_fk_seed)
        self.last_imu_time: Optional[rospy.Time] = None
        self._rate_tracker = RollingRateTracker(self.rate_window_sec)

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=20)
        self.fk_pose_pub = rospy.Publisher(self.fk_pose_topic, PoseStamped, queue_size=20)
        self.cable_sub = message_filters.Subscriber(self.cable_topic, CableLengthsStamped)
        if self.rpy_from_imu:
            self.rpy_sub = message_filters.Subscriber(self.imu_topic, Imu)
        else:
            self.rpy_sub = message_filters.Subscriber(self.mocap_topic, PoseStamped)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rpy_sub, self.cable_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo("CDPR Euler-EKF node started.")
        rospy.loginfo(
            "CDPR kinematics: is_calibrated=%s use_calibrated_cable_length=%s calibration_file=%s",
            str(self.is_calibrated),
            str(self.use_calibrated_cable_length),
            str(self.calibration_file) if self.is_calibrated else "(n/a)",
        )
        rospy.loginfo(
            "Subscribe RPY source: %s (%s), cable lengths: %s (approx sync)",
            self.imu_topic if self.rpy_from_imu else self.mocap_topic,
            "imu" if self.rpy_from_imu else "mocap",
            self.cable_topic,
        )
        rospy.loginfo("Approx sync config: queue_size=%d, slop=%.4f s", self.sync_queue_size, self.sync_slop)
        rospy.loginfo("Publish pose: %s", self.pose_topic)
        rospy.loginfo("Publish fk pose: %s", self.fk_pose_topic)
        if self.fk_with_given_rpy:
            prior_desc = (
                f"pos_prior={self.fk_prior_pos_weight:.3f} (att prior n/a)"
                if self.fk_use_prior
                else "prior off"
            )
        else:
            prior_desc = (
                f"pos_prior={self.fk_prior_pos_weight:.3f}, att_prior={self.fk_prior_att_weight:.3f}"
                if self.fk_use_prior
                else "prior off"
            )
        rospy.loginfo("fk_use_prior=%s, %s", str(self.fk_use_prior), prior_desc)
        rospy.loginfo("fk_with_given_rpy=%s, rpy_from_imu=%s", str(self.fk_with_given_rpy), str(self.rpy_from_imu))
        rospy.loginfo(
            "EKF rate log: log_ekf_rate=%s window=%.1fs period=%.1fs",
            str(self.log_ekf_rate),
            self.rate_window_sec,
            self.rate_log_period,
        )
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

    def _log_ekf_rate_throttled(self, stamp_sec: float) -> None:
        if not self.log_ekf_rate:
            return
        self._rate_tracker.record(stamp_sec)
        snap = self._rate_tracker.snapshot()
        if snap is None:
            return
        period = max(self.rate_log_period, 0.5)
        inst_part = ""
        if snap["inst_hz"] is not None:
            inst_part = (
                f" inst={snap['inst_hz']:.2f}Hz"
                f" dt={snap['dt_mean']*1e3:.2f}±{snap['dt_std']*1e3:.2f}ms"
                f" [{snap['dt_min']*1e3:.2f},{snap['dt_max']*1e3:.2f}]ms"
            )
        rospy.loginfo_throttle(
            period,
            "EKF rate (%.1fs window, n=%d): %.2f Hz%s | ekf.dt=%.4fs",
            snap["span_s"],
            snap["n"],
            snap["rate_hz"],
            inst_part,
            self.ekf.dt,
        )

    def _wait_valid_mocap_init_pose(self) -> Tuple[float, float, float, np.ndarray]:
        """Block until CDPR's mocap callback has a non-degenerate quaternion, or timeout."""
        rate = rospy.Rate(20.0)
        deadline = rospy.Time.now().to_sec() + self.mocap_init_timeout
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        while not rospy.is_shutdown():
            x, y, z, quat = self.cdpr.get_moving_platform_pose_from_mocap()
            q = np.asarray(quat, dtype=float).reshape(4)
            if _quat_valid(q):
                rospy.loginfo("Mocap pose received for EKF init (valid quaternion).")
                return float(x), float(y), float(z), q
            if rospy.Time.now().to_sec() > deadline:
                rospy.logwarn(
                    "No valid mocap quaternion within %.1f s (VRPN may be down or not streaming); "
                    "using identity orientation for EKF init. Position still from last mocap message.",
                    self.mocap_init_timeout,
                )
                return float(x), float(y), float(z), identity.copy()
            rate.sleep()
        rospy.logwarn("Shutdown before valid mocap; EKF init uses identity orientation.")
        x, y, z, _ = self.cdpr.get_moving_platform_pose_from_mocap()
        return float(x), float(y), float(z), identity.copy()

    def synced_callback(self, rpy_msg, cable_msg: CableLengthsStamped) -> None:
        arr = np.asarray(cable_msg.lengths, dtype=float)
        if arr.size != self.geom.m:
            rospy.logwarn_throttle(
                2.0,
                "Received cable length size %d, expected %d. Ignore this message.",
                arr.size,
                self.geom.m,
            )
            return
        now = rpy_msg.header.stamp if rpy_msg.header.stamp != rospy.Time() else rospy.Time.now()
        if self.last_imu_time is not None:
            dt = (now - self.last_imu_time).to_sec()
            if dt > 1e-5 and math.isfinite(dt):
                self.ekf.dt = dt
            else:
                self.ekf.dt = self.default_dt
        self.last_imu_time = now

        if self.rpy_from_imu:
            u1 = self._correct_imu_vector(
                np.array(
                    [
                        rpy_msg.linear_acceleration.x,
                        rpy_msg.linear_acceleration.y,
                        rpy_msg.linear_acceleration.z,
                    ],
                    dtype=float,
                )
            )
            u2 = self._correct_imu_vector(
                np.array(
                    [
                        rpy_msg.angular_velocity.x,
                        rpy_msg.angular_velocity.y,
                        rpy_msg.angular_velocity.z,
                    ],
                    dtype=float,
                )
            )
        else:
            u1 = np.zeros(3, dtype=float)
            u2 = np.zeros(3, dtype=float)

        self.ekf.predict(u1, u2)

        if self.fk_with_given_rpy:
            if self.rpy_from_imu:
                quat = self._correct_imu_quat(
                    np.array(
                        [
                            rpy_msg.orientation.x,
                            rpy_msg.orientation.y,
                            rpy_msg.orientation.z,
                            rpy_msg.orientation.w,
                        ],
                        dtype=float,
                    )
                )
            else:
                quat = np.array(
                    [
                        rpy_msg.pose.orientation.x,
                        rpy_msg.pose.orientation.y,
                        rpy_msg.pose.orientation.z,
                        rpy_msg.pose.orientation.w,
                    ],
                    dtype=float,
                )
            if np.linalg.norm(quat) > 1e-9 and np.isfinite(quat).all():
                yaw, pitch, roll = R.from_quat(quat).as_euler("ZYX", degrees=False)
                theta_given = np.array([roll, pitch, yaw], dtype=float)
            else:
                theta_given = self.ekf.x[6:9].copy()

            if self.fk_use_prior:
                pos_prior = np.array(
                    [self.fk_prior_pos_weight] * 3,
                    dtype=float,
                )
                r_fk = forward_kinematics_lm_xyz_with_fixed_attitude_and_prior(
                    r0=self.r_fk_seed,
                    theta_ba=theta_given,
                    lengths=arr,
                    geom=self.geom,
                    r_prior=self.r_fk_seed,
                    prior_weights=pos_prior,
                    max_iters=self.fk_max_iters,
                )
            else:
                r_fk = forward_kinematics_lm_xyz_with_fixed_attitude(
                    r0=self.r_fk_seed,
                    theta_ba=theta_given,
                    lengths=arr,
                    geom=self.geom,
                    max_iters=self.fk_max_iters,
                )
            self.r_fk_seed = r_fk.copy()
            rho_fk = np.hstack([r_fk, theta_given])
        else:
            if self.fk_use_prior:
                prior_weights = np.array(
                    [
                        self.fk_prior_pos_weight,
                        self.fk_prior_pos_weight,
                        self.fk_prior_pos_weight,
                        self.fk_prior_att_weight,
                        self.fk_prior_att_weight,
                        self.fk_prior_att_weight,
                    ],
                    dtype=float,
                )
                rho_fk = forward_kinematics_lm_with_prior(
                    rho0=self.rho_fk_seed,
                    lengths=arr,
                    geom=self.geom,
                    rho_prior=self.rho_fk_seed,
                    prior_weights=prior_weights,
                    max_iters=self.fk_max_iters,
                )
            else:
                rho_fk = forward_kinematics_lm(
                    rho0=self.rho_fk_seed,
                    lengths=arr,
                    geom=self.geom,
                    max_iters=self.fk_max_iters,
                )
            self.r_fk_seed = rho_fk[0:3].copy()

        self.rho_fk_seed = rho_fk.copy()

        self.ekf.update_with_fk(rho_fk)

        self.publish_fk_pose(now, rho_fk)
        self.publish_pose(now)
        self._log_ekf_rate_throttled(now.to_sec())

    def publish_fk_pose(self, stamp: rospy.Time, rho_fk: np.ndarray) -> None:
        quat = R.from_euler("ZYX", [float(rho_fk[5]), float(rho_fk[4]), float(rho_fk[3])], degrees=False).as_quat()
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(rho_fk[0])
        msg.pose.position.y = float(rho_fk[1])
        msg.pose.position.z = float(rho_fk[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.fk_pose_pub.publish(msg)

    def publish_pose(self, stamp: rospy.Time) -> None:
        x = self.ekf.x
        quat = R.from_euler("ZYX", [float(x[8]), float(x[7]), float(x[6])], degrees=False).as_quat()

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x[0])
        msg.pose.position.y = float(x[1])
        msg.pose.position.z = float(x[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(msg)


def main() -> None:
    rospy.init_node("cdpr_euler_ekf_node", anonymous=False)
    _ = CDPREulerEkfNode()
    rospy.spin()


if __name__ == "__main__":
    main()
