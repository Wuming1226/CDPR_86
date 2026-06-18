#!/usr/bin/env python3

import math

import matplotlib.pyplot as plt
import numpy as np
import rospy
from cdpr_86_msgs.msg import CableLengthsStamped
from cdpr_euler_ekf import (
    CDPRGeometry,
    forward_kinematics_lm,
    forward_kinematics_lm_with_prior,
    wrap_angle,
)
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


def pose_to_xyzrpy(msg: PoseStamped) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    yaw, pitch, roll = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("ZYX", degrees=False)
    return np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=float)


def make_static_geometry() -> CDPRGeometry:
    # Use fixed geometry directly to avoid calling CDPR() inside make_demo_geometry(),
    # which can block on runtime ROS dependencies during standalone tests.
    winches = np.array(
        [
            [-0.260, -0.243, 2.300],
            [-0.361, -0.125, 2.300],
            [-2.049, -0.089, 2.300],
            [-2.169, -0.212, 2.300],
            [-2.193, -1.225, 2.290],
            [-2.084, -1.357, 2.300],
            [-0.415, -1.384, 2.300],
            [-0.290, -1.252, 2.300],
        ],
        dtype=float,
    )
    att = np.array(
        [
            [0.184, -0.125, 0.110],
            [-0.140, 0.169, -0.110],
            [0.140, 0.169, 0.110],
            [-0.184, -0.125, -0.110],
            [-0.184, 0.125, 0.110],
            [0.140, -0.169, -0.110],
            [-0.140, -0.169, 0.110],
            [0.184, 0.125, -0.110],
        ],
        dtype=float,
    )
    return CDPRGeometry(winches_a=winches, attachments_b=att)


def main() -> None:
    rospy.init_node("fk_static_test", anonymous=False)

    pose_topic = rospy.get_param("~pose_topic", "/vrpn_client_node/cdpr/pose")
    cable_topic = rospy.get_param("~cable_topic", "/cable_lengths_measure")
    test_duration_s = float(rospy.get_param("~test_duration_s", 15.0))
    fk_use_prior = bool(rospy.get_param("~fk_use_prior", True))
    fk_max_iters = int(rospy.get_param("~fk_max_iters", 15))
    fk_prior_pos_weight = float(rospy.get_param("~fk_prior_pos_weight", 2.0))
    fk_prior_att_weight = float(rospy.get_param("~fk_prior_att_weight", 40.0))

    rospy.loginfo("FK static test start.")
    rospy.loginfo("pose_topic=%s, cable_topic=%s", pose_topic, cable_topic)
    rospy.loginfo(
        "test_duration_s=%.2f, fk_use_prior=%s, fk_max_iters=%d",
        test_duration_s,
        str(fk_use_prior),
        fk_max_iters,
    )

    # Lock a fixed reference pose at test start.
    pose_msg = rospy.wait_for_message(pose_topic, PoseStamped, timeout=5.0)
    ref_pose = pose_to_xyzrpy(pose_msg)
    rospy.loginfo(
        "Reference pose locked: xyz=[%.4f, %.4f, %.4f], rpy_deg=[%.2f, %.2f, %.2f]",
        ref_pose[0],
        ref_pose[1],
        ref_pose[2],
        math.degrees(ref_pose[3]),
        math.degrees(ref_pose[4]),
        math.degrees(ref_pose[5]),
    )
    geom = make_static_geometry()
    rho_fk_seed = ref_pose.copy()
    prior_weights = np.array(
        [
            fk_prior_pos_weight,
            fk_prior_pos_weight,
            fk_prior_pos_weight,
            fk_prior_att_weight,
            fk_prior_att_weight,
            fk_prior_att_weight,
        ],
        dtype=float,
    )
    ts = []
    fk_hist = []

    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()
        if now - t0 >= test_duration_s:
            break
        try:
            cable_msg = rospy.wait_for_message(cable_topic, CableLengthsStamped, timeout=1.0)
        except rospy.ROSException:
            continue
        lengths = np.asarray(cable_msg.lengths, dtype=float)
        if lengths.size != geom.m:
            rospy.logwarn_throttle(2.0, "Cable size mismatch: got %d expect %d", lengths.size, geom.m)
            continue

        if fk_use_prior:
            rho_fk = forward_kinematics_lm_with_prior(
                rho0=rho_fk_seed,
                lengths=lengths,
                geom=geom,
                rho_prior=rho_fk_seed,
                prior_weights=prior_weights,
                max_iters=fk_max_iters,
            )
        else:
            rho_fk = forward_kinematics_lm(
                rho0=rho_fk_seed,
                lengths=lengths,
                geom=geom,
                max_iters=fk_max_iters,
            )
        rho_fk_seed = rho_fk.copy()

        ts.append(now - t0)
        fk_hist.append(rho_fk)

    if len(ts) == 0:
        rospy.logwarn("No FK samples collected. Abort plotting.")
        return

    t_arr = np.array(ts, dtype=float)
    fk_arr = np.array(fk_hist, dtype=float)
    err = fk_arr - ref_pose.reshape(1, 6)
    for i in range(err.shape[0]):
        err[i, 3] = wrap_angle(err[i, 3])
        err[i, 4] = wrap_angle(err[i, 4])
        err[i, 5] = wrap_angle(err[i, 5])

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]"]
    for i, ax in enumerate(axes):
        y_fk = fk_arr[:, i]
        y_ref = np.full_like(y_fk, ref_pose[i])
        if i >= 3:
            y_fk = np.rad2deg(y_fk)
            y_ref = np.rad2deg(y_ref)
        ax.plot(t_arr, y_fk, "m-", linewidth=1.4, label="fk")
        ax.plot(t_arr, y_ref, "k--", linewidth=1.0, label="ref")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("FK Static Test: pose vs fixed reference")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    fig2, axes2 = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes2):
        y_err = err[:, i]
        if i >= 3:
            y_err = np.rad2deg(y_err)
        ax.plot(t_arr, y_err, "r-", linewidth=1.4)
        ax.set_ylabel("err " + labels[i])
        ax.grid(True, alpha=0.3)
    axes2[-1].set_xlabel("time [s]")
    fig2.suptitle("FK Static Test: error to fixed reference")
    fig2.tight_layout(rect=[0, 0.02, 1, 0.97])

    rospy.loginfo("FK static test done. samples=%d", len(t_arr))
    plt.show()


if __name__ == "__main__":
    main()
