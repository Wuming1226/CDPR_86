#! /usr/bin/env python3

import time
import threading
from collections import deque
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from sensor_msgs.msg import Imu

from cdpr import CDPR
from jacobian import get_jacobian
from generate_traject import smooth_p2p


def wait_for_stable_imu_pub(
    imu_topic: str,
    window_sec: float = 1.0,
    min_messages: int = 20,
) -> None:
    """
    Block until IMU appears to publish steadily: at least min_messages callbacks
    with timestamps falling within a sliding window of window_sec seconds.
    """
    times: deque = deque()
    lock = threading.Lock()

    def _cb(_msg: Imu) -> None:
        now = time.time()
        with lock:
            times.append(now)
            while times and now - times[0] > window_sec:
                times.popleft()

    rospy.Subscriber(imu_topic, Imu, _cb, queue_size=500)
    rospy.loginfo(
        "Waiting for stable IMU on %s (need >= %d msgs in %.2f s)...",
        imu_topic,
        min_messages,
        window_sec,
    )
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        with lock:
            if len(times) >= min_messages:
                rospy.loginfo("IMU on %s looks stable (%d msgs in last %.2f s).", imu_topic, len(times), window_sec)
                return
        rate.sleep()


time_str = datetime.now().strftime("%m%d_%H%M")
# folder = '../data/beta/'
folder = 'src/double_cdpr/data/beta2'
pose_ref_save_path = folder + 'pose_ref_' + time_str + '.txt'
pose_save_path = folder + 'pose_' + time_str + '.txt'
euler_save_path = folder + 'euler_' + time_str + '.txt'
# cable_length_save_path = folder + 'cable_length' + file_order + '.txt'
# motor_velo_save_path = folder + 'motor_velo_' + file_order + '.txt'


if __name__ == "__main__":

    cdpr = CDPR(imu_active=True)
    if cdpr.imu_active:
        wait_for_stable_imu_pub(cdpr.imu_topic)

    T = 0.05     # control period
    rate = rospy.Rate(1/T)

    time.sleep(0.5)

    x_r_list, y_r_list, z_r_list = [], [], [],
    x_list, y_list, z_list = [], [], []
    yaw_r_list, pitch_r_list, roll_r_list = [], [], [],
    yaw_list, pitch_list, roll_list = [], [], []
    # cl1_list, cl2_list, cl3_list, cl4_list = [], [], [], []
    pose_list = np.empty((0, 6))
    pose_ref_list = np.empty((0, 6))
    # cable_length_ref_list = np.empty((0, 4))
    # cable_length_list = np.empty((0, 4))
    # motor_velo_list = np.empty((0, 4))

    # traject = np.loadtxt("path_mix.txt")

    # end_to_center_offset_x = -0.933 - (-0.950)
    # end_to_center_offset_y = -0.728 - (-0.660)
    x0, y0, z0, quat0 = cdpr.get_moving_platform_pose_from_mocap()
    start_pos = np.array([x0, y0, z0])
    start_euler = R.from_quat(quat0).as_euler('ZYX', degrees=False)
    start_point = np.concatenate([start_pos, start_euler])
    print(start_point)
    way_point1 = start_point + np.array([0., 0., 0., 0.0, 0.0, 0.0])
    way_point1[3:6] = np.array([0, 0, -0]) / 180 * np.pi
    # way_point1[2] = 0.65
    # way_point1 = np.array([-1.1774, -0.7348, 0.72, 0, 0, 0])
    # way_point1 = np.array([-1.1574, -0.7348, 0.72, 0, 0, 0])
    # way_point1 = np.array([-1.385, -1.014, 0.72, 0, 0, 0])
    # way_point1 = np.array([-1.020 + end_to_center_offset_x, -0.651 + end_to_center_offset_y, 0.72, 0, 0, 0])
    # way_point2 = way_point1 + np.array([-0.3, -0.2, 0.0, 0.0, 0.0, 0.0])
    # way_point3 = way_point2 - np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0])
    # way_point4 = way_point3 + np.array([0.1, 0, 0, 0.0, 0.0, 0.0])
    traject_len = np.linalg.norm(way_point1 - start_point)
    traject = smooth_p2p([start_point, way_point1, way_point1],
                         [3, 2], np.inf, 0.05)
    # traject = smooth_p2p([start_point, start_point],
    #                      [10], np.inf, 0.05)
    # traject = smooth_p2p([start_point, way_point1], [10], np.inf, 0.05)

    # tighten_flag = True

    # ---------------------- main loop ----------------------

    time.sleep(2)
    # cdpr.pretighten(True, True, True, True)
    # cdpr_high.init_cable_length(True, True, True, True)
    # cdpr_low.init_cable_length(True, True, True, True)

    cnt = 0
    lst_err = 0

    # ---------------------- main loop ----------------------

    while not rospy.is_shutdown() and cnt < len(traject):

        print('-----------------------------------------------')
        print('                   run: {}'.format(cnt))

        start_time = time.time()

        # 参考数值（所有数据均在基座坐标系下）
        pos_ref = traject[cnt][0:3]
        euler_ref = traject[cnt][3:6]
        rot_ref = R.from_euler('ZYX', euler_ref)
        pose_ref = np.concatenate([pos_ref, euler_ref])
        # print('{:>16}: {:>8}\t{:>8}'.format('pose_ref:', pose_ref_h, pose_ref_l))
        print('pose_ref: {}'.format(pose_ref))

        if cnt == len(traject) - 1:     # 防溢出
            pos_ref_next = traject[cnt][0:3]
            euler_ref_next = traject[cnt][3:6]
            rot_ref_next = R.from_euler('ZYX', euler_ref_next)
        else:
            pos_ref_next = traject[cnt + 1][0:3]
            euler_ref_next = traject[cnt + 1][3:6]
            rot_ref_next = R.from_euler('ZYX', euler_ref_next)
        # print('{:>16}: {:>8}\t{:>8}'.format('pose_ref_next:', pose_ref_next_h, pose_ref_next_l))
        print('pose_ref_next: {} {}'.format(pos_ref_next, euler_ref_next))

        cnt += 1

        # 实际数值
        x, y, z, quat = cdpr.get_moving_platform_pose_from_mocap()
        pos = np.array([x, y, z])
        rot = R.from_quat(quat)  # 当前姿态
        euler = rot.as_euler('ZYX', degrees=False)
        pose = np.concatenate([pos, euler])

        # print('{:>16}: {:>8}\t{:>8}'.format('pose:', pose_h, pose_l))
        print('pose: {}'.format(np.concatenate([pos, euler * 180 / np.pi])))
        # cable_length = cdpr.get_cable_length()
        # print('cable length: {}'.format(cable_length))

        # 位姿误差
        pos_err = pos_ref - pos
        rot_err = rot_ref * rot.inv()  # R_err = R_ref * R^T
        ori_err = rot_err.as_rotvec()
        # print('{:>16}: {:>8}\t{:>8}'.format('pose_err:', pose_err_h, pose_err_l))
        euler_err = euler_ref - euler
        pose_err = np.hstack((pos_err, ori_err))
        print('pose_err: {}'.format(pose_err))
        print('pose_err(euler): {}'.format(np.hstack((pos_err, euler_err * 180 / np.pi))))

        # --- 参考速度 ---
        v_ref = (pos_ref_next - pos_ref) / T
        rot_delta = rot_ref_next * rot_ref.inv()
        omega_ref = rot_delta.as_rotvec() / T
        velo_ref = np.hstack((v_ref, omega_ref)).reshape(-1, 1)
        print('velo_ref: {}'.format(velo_ref))

        # --- 滑模 ---
        eps = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        eps = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        k = np.diag([1.1, 1.1, 1.1, 2.0, 2.0, 2.0])

        velo_task = (
                velo_ref
                + eps @ np.sign(pose_err.reshape(-1, 1))
                + k @ pose_err.reshape(-1, 1)
        )
        # print('{:>16}: {:>8}\t{:>8}'.format('velo_task:', velo_task_h.flatten(), velo_task_l.flatten()))
        print('velo_task: {}'.format(velo_task.flatten()))

        # 逆运动学
        J = get_jacobian(cdpr.a_matrix, cdpr.b_matrix, pos, quat)
        # print("jacobian: {}".format(J))
        velo_joint = (J @ velo_task).reshape(8, )
        # print('{:>16}: {:>8}\t{:>8}'.format('velo_joint:', velo_joint_h, velo_joint_l))
        print('velo_joint: {}'.format(velo_joint))

        # convert linear velocities to velocities of motors
        velo_motor = velo_joint / 0.025      # 0.025 is radius of the coil

        # set cable velocity in joint space
        velo_limit = 10
        for i, vel in enumerate(velo_motor):
            if np.abs(vel) > velo_limit:      # velocity limit
                velo_motor[i] = velo_limit * np.sign(vel)

        velo_motor[0] = -velo_motor[0]
        velo_motor[2] = -velo_motor[2]
        velo_motor[4] = -velo_motor[4]
        velo_motor[6] = -velo_motor[6]
        cdpr.set_motor_velo(velo_motor)
        print('motor_velo: {}'.format(velo_motor))

        x_r_list.append(pos_ref[0])
        y_r_list.append(pos_ref[1])
        z_r_list.append(pos_ref[2])
        yaw_r_list.append(euler_ref[0])
        pitch_r_list.append(euler_ref[1])
        roll_r_list.append(euler_ref[2])

        x_list.append(pos[0])
        y_list.append(pos[1])
        z_list.append(pos[2])
        yaw_list.append(euler[0])
        pitch_list.append(euler[1])
        roll_list.append(euler[2])

        # cl1_list.append(cable_lengvelo_joint_h, velo_joint_lth[0])
        # cl2_list.append(cable_length[1])
        # cl3_list.append(cable_length[2])
        # cl4_list.append(cable_length[3])

        # data 
        pose_list = np.vstack((pose_list, pose))
        pose_ref_list = np.vstack((pose_ref_list, pose_ref))
        # cable_length_list = np.row_stack((cable_length_list, cable_length))
        # length_controller_list = np.row_stack((length_controller_list, veloJoint1))
        # motor_velo_list = np.vstack((motor_velo_list, velo_motor))

        # np.savetxt(pose_ref_save_path, pose_ref_list)
        # np.savetxt(pose_save_path, pose_list)
        # np.savetxt(cable_length_save_path, cable_length_list)
        # np.savetxt(length_controller_save_path, length_controller_list)
        # np.savetxt(motor_velo_save_path, motor_velo_list)
        print('data saved.')

        end_time = time.time()
        print("loop time: {}".format(end_time - start_time))

        rate.sleep()

    cdpr.set_motor_velo([0, 0, 0, 0, 0, 0, 0, 0])

    # calculate error
    x_e = np.array(x_r_list) - np.array(x_list)
    y_e = np.array(y_r_list) - np.array(y_list)
    z_e = np.array(z_r_list) - np.array(z_list)
    err_arr = np.sqrt(x_e ** 2 + y_e ** 2 + z_e ** 2)
    print("\n\n-----------------------------")
    print("mean tracking error: {}".format(np.mean(err_arr)))
    print("max tracking error: {}".format(np.max(err_arr)))

    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(3, 2, 1)
    y_plot = fig.add_subplot(3, 2, 3)
    z_plot = fig.add_subplot(3, 2, 5)
    yaw_plot = fig.add_subplot(3, 2, 2)
    pitch_plot = fig.add_subplot(3, 2, 4)
    roll_plot = fig.add_subplot(3, 2, 6)
    # c4_plot = fig.add_subplot(4, 2, 8)

    x_plot.plot(x_r_list)
    x_plot.plot(x_list)
    y_plot.plot(y_r_list)
    y_plot.plot(y_list)
    z_plot.plot(z_r_list)
    z_plot.plot(z_list)
    yaw_plot.plot(yaw_r_list)
    yaw_plot.plot(yaw_list)
    pitch_plot.plot(pitch_r_list)
    pitch_plot.plot(pitch_list)
    roll_plot.plot(roll_r_list)
    roll_plot.plot(roll_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    # c1_plot.plot(cl1_list)
    # c2_plot.plot(cl2_list)
    # c3_plot.plot(cl3_list)
    # c4_plot.plot(cl4_list)

    plt.ioff()
    plt.show()


