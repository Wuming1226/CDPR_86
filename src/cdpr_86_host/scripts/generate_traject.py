import numpy as np
import matplotlib.pyplot as plt


def smooth_p2p(way_points_list, travel_time_list, velo_limit, time_step):
    traject = []

    for index, travel_time in enumerate(travel_time_list):
        max_velo = (way_points_list[index+1] - way_points_list[index]) / (0.75 * travel_time)
        if np.linalg.norm(max_velo) > velo_limit:
            return None
        else:
            step_number = travel_time / time_step
            pos = way_points_list[index].copy()     # 真坑啊 nparray
            for step in range(int(step_number)):
                if step < 0.25 * step_number:
                    velo = max_velo * step / (0.25 * step_number)
                elif step < 0.75 * step_number:
                    velo = max_velo
                else:
                    velo = max_velo * (1 - (step - 0.75 * step_number) / (0.25 * step_number))
                pos += velo * time_step
                traject.append(pos.copy())

    return traject


def arc_interpolation(center, p1, p2, T, dt):
    center = np.array(center, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # radius
    r = np.linalg.norm(p1 - center)

    # angles
    ang1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    ang2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])

    # shortest arc
    dtheta = ang2 - ang1
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

    # time stamps
    t_list = np.arange(0, T, dt)

    # quintic time scaling s ∈ [0,1]
    s = quintic_time_scaling(T, t_list)

    # interpolated angle
    theta = ang1 + dtheta * s

    # arc points
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    arc_points = np.vstack((x, y)).T

    return arc_points



def quintic_time_scaling(T, t):
    """
    Quintic polynomial s(t) in [0,1], with:
    s(0)=0, s(T)=1
    s'(0)=s'(T)=0
    s''(0)=s''(T)=0
    """
    tau = t / T
    return 10*tau**3 - 15*tau**4 + 6*tau**5


if __name__ == '__main__':

    # zero_pos = np.array([0, 0, 0.430])
    # target1 = np.array([0.445, 0.210, 0.078]) - np.array([0.249, 0.1515, 0])
    # safe_point1 = np.array([0.200, 0.150, 0.250])  # 安全位置1
    # safe_point2 = np.array([-0.200, -0.150, 0.250])  # 安全位置2
    # safe_point3 = np.array([-0.200, 0.150, 0.250])
    # traject_height = -0.050 + 0.172  # 轨迹高度（实测）
    # traject_1 = np.array([0.153, 0.000, traject_height])
    # traject_2 = np.array([0.153, -0.153, traject_height])
    # traject_3 = np.array([-0.160, -0.153, traject_height])
    # traject_4 = np.array([-0.160, 0.153, traject_height])
    #
    # waypoints = [zero_pos, safe_point1, target1 + np.array([0, 0, 0.050]), target1, target1, traject_1,
    #              traject_2, traject_3, traject_4, traject_4, safe_point3, zero_pos]
    # travel_time = [10, 8, 6, 6, 10, 15, 30, 30, 6, 8, 10]
    # velo_limit = 0.100
    # time_step = 0.1
    #
    # traject = smooth_p2p(waypoints, travel_time, velo_limit, time_step)
    # traject = np.array(traject)
    #
    # np.savetxt("trajectory+.txt", traject)

    way_point11 = np.array([-3e-01, -3e-01, 3.2e-01])
    way_point12 = np.array([3e-01, -3e-01, 2e-01])
    arc1 = arc_interpolation(np.array([0, 0]), np.array([-3e-01, 3e-01]),
                              np.array([-3e-01, -3e-01]), 20, 0.1)
    arc1 = np.hstack([arc1, np.ones([200, 1]) * 3.2e-01])
    line1 = smooth_p2p([np.array([3e-01, 3e-01, 2e-01]), np.array([3e-01, -3e-01, 2e-01])],
                       [20], np.inf, 0.1)
    line1 = np.array(line1)
    arc2 = arc_interpolation(np.array([0, 0]), np.array([3e-01, 3e-01]),
                              np.array([3e-01, -3e-01]), 20, 0.1)
    arc2 = np.hstack([arc2, np.ones([200, 1]) * 3.2e-01])
    line2 = smooth_p2p([np.array([-3e-01, 3e-01, 2e-01]), np.array([-3e-01, -3e-01, 2e-01])]
                       , [20], np.inf, 0.1)
    line2 = np.array(line2)

    switch = np.loadtxt("path_x.txt")

    print(arc1.shape)
    print(line1.shape)

    traject1 = np.hstack([arc1, line1])
    traject2 = np.hstack([arc2, line2])

    traject = np.vstack([traject1, switch, traject2])

    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(4, 2, 1)
    y_plot = fig.add_subplot(4, 2, 3)
    z_plot = fig.add_subplot(4, 2, 5)

    x_plot.plot(traject[:, 0])
    y_plot.plot(traject[:, 1])
    z_plot.plot(traject[:, 2])

    x_plot = fig.add_subplot(4, 2, 2)
    y_plot = fig.add_subplot(4, 2, 4)
    z_plot = fig.add_subplot(4, 2, 6)

    x_plot.plot(traject[:, 3])
    y_plot.plot(traject[:, 4])
    z_plot.plot(traject[:, 5])

    plt.ioff()
    plt.show()

    np.savetxt("path_mix.txt", traject)

