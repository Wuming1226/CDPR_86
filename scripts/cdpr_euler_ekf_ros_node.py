#!/usr/bin/env python3
import math
from typing import Optional

import message_filters
import numpy as np
import rospy
from cdpr_86_host.msg import CableLengthsStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R

from cdpr import CDPR
from cdpr_euler_ekf import (
    EulerEKFCDPR,
    forward_kinematics_lm,
    forward_kinematics_lm_with_prior,
    forward_kinematics_lm_xyz_with_fixed_attitude,
    make_demo_geometry,
)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class CDPREulerEkfNode:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.pose_topic = rospy.get_param("~pose_topic", "/ekf_pose")
        self.fk_pose_topic = rospy.get_param("~fk_pose_topic", "/fk_pose")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.cable_topic = rospy.get_param("~cable_topic", "/cable_lengths_measure")
        self.default_dt = float(rospy.get_param("~default_dt", 0.01))
        self.fk_max_iters = int(rospy.get_param("~fk_max_iters", 20))
        self.fk_use_prior = _as_bool(rospy.get_param("~fk_use_prior", True))
        self.fk_prior_pos_weight = float(rospy.get_param("~fk_prior_pos_weight", 5.0))
        self.fk_prior_att_weight = float(rospy.get_param("~fk_prior_att_weight", 20.0))
        self.fk_xyz_only_with_imu_rpy = _as_bool(rospy.get_param("~fk_xyz_only_with_imu_rpy", False))
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 100))
        self.sync_slop = float(rospy.get_param("~sync_slop", 1.0 / 15))

        g_a = np.array([0.0, 0.0, -9.81], dtype=float)
        self.geom = make_demo_geometry()
        self.ekf = EulerEKFCDPR(dt=self.default_dt, g_a=g_a)
        self.cdpr = CDPR(imu_active=True)

        x0 = np.zeros(15, dtype=float)
        x, y, z, quat = self.cdpr.get_moving_platform_pose_from_mocap()
        yaw, pitch, roll = R.from_quat(quat).as_euler("ZYX", degrees=False)
        x0[0:3] = np.array([x, y, z], dtype=float)
        x0[6:9] = np.array([roll, pitch, yaw], dtype=float)
        p0 = np.eye(15, dtype=float) * 0.03
        p0[6:9, 6:9] *= 4.0
        self.ekf.set_initial(x0, p0)

        self.rho_fk_seed = np.hstack([self.ekf.x[0:3], self.ekf.x[6:9]])
        self.r_fk_seed = self.ekf.x[0:3].copy()
        print(self.rho_fk_seed)
        self.last_imu_time: Optional[rospy.Time] = None

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=20)
        self.fk_pose_pub = rospy.Publisher(self.fk_pose_topic, PoseStamped, queue_size=20)
        self.imu_sub = message_filters.Subscriber(self.imu_topic, Imu)
        self.cable_sub = message_filters.Subscriber(self.cable_topic, CableLengthsStamped)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.imu_sub, self.cable_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo("CDPR Euler-EKF node started.")
        rospy.loginfo("Subscribe IMU: %s, cable lengths: %s (approx sync)", self.imu_topic, self.cable_topic)
        rospy.loginfo("Approx sync config: queue_size=%d, slop=%.4f s", self.sync_queue_size, self.sync_slop)
        rospy.loginfo("Publish pose: %s", self.pose_topic)
        rospy.loginfo("Publish fk pose: %s", self.fk_pose_topic)
        rospy.loginfo(
            "fk_use_prior=%s, fk_prior_pos_weight=%.3f, fk_prior_att_weight=%.3f",
            str(self.fk_use_prior),
            self.fk_prior_pos_weight,
            self.fk_prior_att_weight,
        )
        rospy.loginfo("fk_xyz_only_with_imu_rpy=%s", str(self.fk_xyz_only_with_imu_rpy))

    def synced_callback(self, imu_msg: Imu, cable_msg: CableLengthsStamped) -> None:
        arr = np.asarray(cable_msg.lengths, dtype=float)
        if arr.size != self.geom.m:
            rospy.logwarn_throttle(
                2.0,
                "Received cable length size %d, expected %d. Ignore this message.",
                arr.size,
                self.geom.m,
            )
            return
        now = imu_msg.header.stamp if imu_msg.header.stamp != rospy.Time() else rospy.Time.now()
        if self.last_imu_time is not None:
            dt = (now - self.last_imu_time).to_sec()
            if dt > 1e-5 and math.isfinite(dt):
                self.ekf.dt = dt
            else:
                self.ekf.dt = self.default_dt
        self.last_imu_time = now

        u1 = np.array(
            [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ],
            dtype=float,
        )
        u2 = np.array(
            [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z],
            dtype=float,
        )

        self.ekf.predict(u1, u2)

        if self.fk_xyz_only_with_imu_rpy:
            imu_quat = np.array(
                [
                    imu_msg.orientation.x,
                    imu_msg.orientation.y,
                    imu_msg.orientation.z,
                    imu_msg.orientation.w,
                ],
                dtype=float,
            )
            if np.linalg.norm(imu_quat) > 1e-9 and np.isfinite(imu_quat).all():
                yaw, pitch, roll = R.from_quat(imu_quat).as_euler("ZYX", degrees=False)
                theta_imu = np.array([roll, pitch, yaw], dtype=float)
            else:
                theta_imu = self.ekf.x[6:9].copy()

            r_fk = forward_kinematics_lm_xyz_with_fixed_attitude(
                r0=self.r_fk_seed,
                theta_ba=theta_imu,
                lengths=arr,
                geom=self.geom,
                max_iters=self.fk_max_iters,
            )
            self.r_fk_seed = r_fk.copy()
            rho_fk = np.hstack([r_fk, theta_imu])
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
