#! /usr/bin/env python3

import time
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist

from cdpr import CDPR
from jacobian import get_jacobian


class JoystickClosedLoopController:
    def __init__(self):
        self.cdpr = CDPR(imu_active=False, is_calibrated=True, calibration_file="cdpr_kinematic_calib.json")

        self.control_period = rospy.get_param("~control_period", 1.0 / 15)
        self.coil_radius = rospy.get_param("~coil_radius", 0.025)
        self.velo_limit = rospy.get_param("~motor_velo_limit", 10.0)
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.5)
        self.vel_filter_alpha = rospy.get_param("~vel_filter_alpha", 0.3)
        self.linear_acc_limit = rospy.get_param("~linear_acc_limit", 1.0)
        self.angular_acc_limit = rospy.get_param("~angular_acc_limit", 2.0)

        # Position/orientation feedback gains in task space.
        self.k = np.diag(rospy.get_param("~k_diag", [1.1, 1.1, 1.1, 2.0, 2.0, 2.0]))

        self.cmd_vel = np.zeros(6)
        self.last_cmd_time = None

        self.pos_ref = None
        self.rot_ref = None
        self.filtered_cmd_vel = np.zeros(6)
        self.smoothed_cmd_vel = np.zeros(6)

        rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback, queue_size=1)

    def cmd_vel_callback(self, msg):
        self.cmd_vel = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        ], dtype=float)
        print(self.cmd_vel)
        self.last_cmd_time = rospy.Time.now()

    def get_cmd_velocity(self):
        if self.last_cmd_time is None:
            return np.zeros(6)

        if (rospy.Time.now() - self.last_cmd_time).to_sec() > self.cmd_timeout:
            return np.zeros(6)

        return self.cmd_vel.copy()

    def smooth_cmd_velocity(self, raw_cmd_vel):
        alpha = np.clip(self.vel_filter_alpha, 0.0, 1.0)
        self.filtered_cmd_vel = alpha * raw_cmd_vel + (1.0 - alpha) * self.filtered_cmd_vel

        max_dv = np.array([
            self.linear_acc_limit,
            self.linear_acc_limit,
            self.linear_acc_limit,
            self.angular_acc_limit,
            self.angular_acc_limit,
            self.angular_acc_limit,
        ], dtype=float) * self.control_period

        dv = self.filtered_cmd_vel - self.smoothed_cmd_vel
        dv = np.clip(dv, -max_dv, max_dv)
        self.smoothed_cmd_vel = self.smoothed_cmd_vel + dv

        return self.smoothed_cmd_vel.copy()

    def update_reference_pose(self, cmd_vel):
        # Integrate cmd_vel into target pose (online trajectory generation).
        self.pos_ref = self.pos_ref + cmd_vel[:3] * self.control_period
        rot_inc = R.from_rotvec(cmd_vel[3:] * self.control_period)
        self.rot_ref = rot_inc * self.rot_ref

    def spin(self):
        rate = rospy.Rate(1.0 / self.control_period)
        time.sleep(0.5)

        # Initialize target pose as current pose.
        x, y, z, quat = self.cdpr.get_moving_platform_pose_from_mocap()
        self.pos_ref = np.array([x, y, z], dtype=float)
        self.rot_ref = R.from_quat(quat)

        while not rospy.is_shutdown():
            x, y, z, quat = self.cdpr.get_moving_platform_pose_from_mocap()
            pos = np.array([x, y, z], dtype=float)
            rot = R.from_quat(quat)

            cmd_vel_raw = self.get_cmd_velocity()
            cmd_vel = self.smooth_cmd_velocity(cmd_vel_raw)
            self.update_reference_pose(cmd_vel)

            pos_err = self.pos_ref - pos
            rot_err = self.rot_ref * rot.inv()
            ori_err = rot_err.as_rotvec()
            pose_err = np.hstack((pos_err, ori_err)).reshape(-1, 1)

            velo_ref = cmd_vel.reshape(-1, 1)
            velo_task = velo_ref + self.k @ pose_err

            J = get_jacobian(self.cdpr.a_matrix, self.cdpr.b_matrix, pos, quat)
            velo_joint = (J @ velo_task).reshape(8,)

            velo_motor = velo_joint / self.coil_radius
            velo_motor = np.clip(velo_motor, -self.velo_limit, self.velo_limit)

            velo_motor[0] = -velo_motor[0]
            velo_motor[2] = -velo_motor[2]
            velo_motor[4] = -velo_motor[4]
            velo_motor[6] = -velo_motor[6]

            self.cdpr.set_motor_velo(velo_motor)
            rate.sleep()

        self.cdpr.set_motor_velo(np.zeros(8))


if __name__ == "__main__":
    controller = JoystickClosedLoopController()
    try:
        controller.spin()
    finally:
        controller.cdpr.set_motor_velo(np.zeros(8))
