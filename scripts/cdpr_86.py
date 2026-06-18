#! /usr/bin/env python3

import os
import rospy
import time
import numpy as np
import can
import threading

from motor import Motor
from codec import le_bytes_to_int

from std_msgs.msg import Float32MultiArray
from cdpr_86_msg.msg import MotorPositionsStamped
#from slave.srv import SetMotorVelo, SetMotorVeloResponse, GetMotorPos, GetMotorPosResponse


class CDPR:
    def __init__(self, motor1_id=1, motor2_id=2, motor3_id=3, motor4_id=4, motor5_id=5, motor6_id=6, motor7_id=7, motor8_id=8):

        # can settings
        os.system('sudo ip link set can0 type can bitrate 1000000 restart-ms 100')
        os.system('sudo ip link set can0 txqueuelen 10000')
        os.system('sudo ip link set can0 up')
        
        self.can_channel = 'can0'
        
        # ros settings
        rospy.init_node('cdpr_86', anonymous=True)
        rospy.Subscriber('motor_velo', Float32MultiArray, callback=self.velo_callback)
        self.motor_pos_rel_pub = rospy.Publisher('motor_pos_rel', MotorPositionsStamped, queue_size=10)
        self.motor_pos_abs_pub = rospy.Publisher('motor_pos_abs', MotorPositionsStamped, queue_size=10)
        
        # safety settings
        self.last_velo_callback_time = time.time()
        self.max_interval = 0.5

        # initialize motors
        self._motor1 = Motor(self.can_channel, motor1_id)
        self._motor1.set_profile_velocity_mode(0)
        self._motor2 = Motor(self.can_channel, motor2_id)
        self._motor2.set_profile_velocity_mode(0)
        self._motor3 = Motor(self.can_channel, motor3_id)
        self._motor3.set_profile_velocity_mode(0)
        self._motor4 = Motor(self.can_channel, motor4_id)
        self._motor4.set_profile_velocity_mode(0)
        self._motor5 = Motor(self.can_channel, motor5_id)
        self._motor5.set_profile_velocity_mode(0)
        self._motor6 = Motor(self.can_channel, motor6_id)
        self._motor6.set_profile_velocity_mode(0)
        self._motor7 = Motor(self.can_channel, motor7_id)
        self._motor7.set_profile_velocity_mode(0)
        self._motor8 = Motor(self.can_channel, motor8_id)
        self._motor8.set_profile_velocity_mode(0)
        
    def set_velocity(self, velo):
        self._motor1.set_target_velocity(velo[0])
        self._motor2.set_target_velocity(velo[1])
        self._motor3.set_target_velocity(velo[2])
        self._motor4.set_target_velocity(velo[3])
        self._motor5.set_target_velocity(velo[4])
        self._motor6.set_target_velocity(velo[5])
        self._motor7.set_target_velocity(velo[6])
        self._motor8.set_target_velocity(velo[7])
        
    def shut_down(self):
        self._motor1.shut_down()
        self._motor2.shut_down()
        self._motor3.shut_down()
        self._motor4.shut_down()
        self._motor5.shut_down()
        self._motor6.shut_down()
        self._motor7.shut_down()
        self._motor8.shut_down()

        
    def velo_callback(self, msg):
        rospy.loginfo_throttle(1.0, "target velocity: %s", list(msg.data))
        self.last_velo_callback_time = time.time()
        self.set_velocity(msg.data)
        
    # def pub_motor_pos(self):
    #     motor_pos = Int64MultiArray(data=np.array([self._motor1.getPos(), self._motor2.getPos()]))
    #     self._motor_pos_pub.publish(motor_pos)
    # def receiveMotorVelo(self):
    #     msg = rospy.wait_for_message('motor_velo', Int16MultiArray, timeout=None)
    #     self._motor1.setVelo(msg.data[0])
    #     self._motor2.setVelo(msg.data[1])
    #     self._motor3.setVelo(msg.data[2])
    #     print(msg.data[0], msg.data[1], msg.data[2])
    # def _setMotorVeloHandle(self, req):
    #     self._motor1.setVelo(req.motor1Velo)
    #     self._motor2.setVelo(req.motor2Velo)
    #     self._motor3.setVelo(req.motor3Velo)
    #     print(req.motor1Velo, req.motor2Velo, req.motor3Velo)
    #     return SetMotorVeloResponse(True)



if __name__=="__main__":
    cdpr = CDPR()

    sync_hz = 200.0
    sync_period = 1.0 / sync_hz
    pdo_window = 0.004  # TPDO should arrive shortly after each SYNC
    init_delay = 0.5   # wait for system to settle before latching init position

    bus = can.interface.Bus(channel='can0', bustype='socketcan')
    tpdo_ids = [0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287, 0x288]
    id_to_idx = {can_id: idx for idx, can_id in enumerate(tpdo_ids)}
    bus.set_filters([{"can_id": can_id, "can_mask": 0x7FF, "extended": False} for can_id in tpdo_ids])
    sync_msg = can.Message(arbitration_id=0x080, data=[0x01], is_extended_id=False)
    running = threading.Event()
    running.set()
    state_lock = threading.Lock()

    latest_pos = np.zeros(8, dtype=np.int64)
    state = {
        "init_motor_pos": None,
        "start_time": time.monotonic(),
        "current_sync_seq": 0,
        "last_sync_time": time.monotonic(),
        "total_cycles": 0,
        "complete_cycles": 0,
        "timeout_stopped": False,
        "stats_last_time": time.monotonic(),
    }
    last_rx_sync_seq = np.full(8, -1, dtype=np.int64)
    last_rx_arrival_time = np.full(8, -1.0, dtype=np.float64)
    miss_per_motor = np.zeros(8, dtype=np.int64)

    # Map monotonic SYNC times to ROS wall time for header.stamp.
    clock_sync = {
        "ros0": rospy.Time.now().to_sec(),
        "mono0": time.monotonic(),
    }

    def mono_to_ros_stamp(mono_t: float) -> rospy.Time:
        return rospy.Time.from_sec(clock_sync["ros0"] + (mono_t - clock_sync["mono0"]))

    def make_motor_pos_msg(positions, sync_mono_time: float) -> MotorPositionsStamped:
        msg = MotorPositionsStamped()
        msg.header.stamp = mono_to_ros_stamp(sync_mono_time)
        msg.header.frame_id = "cdpr_motor_encoder"
        msg.positions = positions.astype(np.float32).tolist()
        return msg

    def publish_complete_cycle(cycle_time: float, pos_snapshot: np.ndarray, init_pos_snapshot):
        cdpr.motor_pos_abs_pub.publish(
            make_motor_pos_msg(pos_snapshot, cycle_time)
        )
        if init_pos_snapshot is not None:
            motor_pos_rel = (pos_snapshot - init_pos_snapshot) / cdpr._motor1.encoder_resolution * 2 * np.pi
            cdpr.motor_pos_rel_pub.publish(
                make_motor_pos_msg(motor_pos_rel, cycle_time)
            )

    def tx_sync_loop():
        nonlocal_vars = {
            "next_sync_time": time.monotonic()
        }
        while running.is_set() and not rospy.is_shutdown():
            now = time.monotonic()
            if now < nonlocal_vars["next_sync_time"]:
                time.sleep(min(0.001, nonlocal_vars["next_sync_time"] - now))
                continue

            with state_lock:
                cycle_seq = state["current_sync_seq"]
                cycle_time = state["last_sync_time"]

            # Evaluate the current cycle before sending the next SYNC, so fast TPDOs
            # from the upcoming SYNC cannot overwrite last_rx_sync_seq first.
            if cycle_seq > 0:
                with state_lock:
                    state["total_cycles"] += 1
                    missing = []
                    for idx in range(8):
                        got_same_seq = last_rx_sync_seq[idx] == cycle_seq
                        got_in_window = got_same_seq and (0.0 <= (last_rx_arrival_time[idx] - cycle_time) <= pdo_window)
                        if not got_in_window:
                            missing.append(idx)
                    if not missing:
                        state["complete_cycles"] += 1
                        pos_snapshot = latest_pos.astype(np.float64).copy()
                        if state["init_motor_pos"] is None:
                            startup_ready = (now - state["start_time"]) >= init_delay
                            if startup_ready:
                                state["init_motor_pos"] = pos_snapshot.copy()
                                rospy.loginfo(
                                    "Initial motor positions latched from synchronized TPDO cycle %d after %.2fs startup delay",
                                    cycle_seq,
                                    init_delay,
                                )
                        init_pos_snapshot = state["init_motor_pos"]
                    else:
                        for idx in missing:
                            miss_per_motor[idx] += 1
                        pos_snapshot = None
                        init_pos_snapshot = None
                    total_snapshot = state["total_cycles"]
                    complete_snapshot = state["complete_cycles"]
                    miss_snapshot = miss_per_motor.copy()

                if pos_snapshot is not None:
                    publish_complete_cycle(cycle_time, pos_snapshot, init_pos_snapshot)

                if now - state["stats_last_time"] >= 1.0:
                    state["stats_last_time"] = now
                    if total_snapshot > 0:
                        ratio = 100.0 * complete_snapshot / total_snapshot
                        rospy.loginfo(
                            "PDO cycles complete=%d/%d (%.1f%%), miss_per_motor=%s",
                            complete_snapshot, total_snapshot, ratio, miss_snapshot.tolist()
                        )

            try:
                bus.send(sync_msg)
            except can.CanError as e:
                rospy.logwarn("SYNC send failed: %s", e)

            with state_lock:
                state["current_sync_seq"] = cycle_seq + 1
                state["last_sync_time"] = time.monotonic()

            if time.time() - cdpr.last_velo_callback_time > cdpr.max_interval:
                if not state["timeout_stopped"]:
                    cdpr.set_velocity([0, 0, 0, 0, 0, 0, 0, 0])
                    state["timeout_stopped"] = True
                    rospy.logwarn("No velocity command, force stop all motors")
            else:
                state["timeout_stopped"] = False

            nonlocal_vars["next_sync_time"] += sync_period

    def rx_pdo_loop():
        while running.is_set() and not rospy.is_shutdown():
            msg = bus.recv(timeout=0.01)
            if msg is None:
                continue
            idx = id_to_idx.get(msg.arbitration_id)
            if idx is None:
                continue
            if len(msg.data) < 4:
                rospy.logwarn("Invalid TPDO length from 0x%03X: %d", msg.arbitration_id, len(msg.data))
                continue

            pos = le_bytes_to_int(msg.data[4:8])
            recv_time = time.monotonic()
            with state_lock:
                latest_pos[idx] = pos
                last_rx_sync_seq[idx] = state["current_sync_seq"]
                last_rx_arrival_time[idx] = recv_time

    try:
        tx_thread = threading.Thread(target=tx_sync_loop, name="tx_sync_loop", daemon=True)
        rx_thread = threading.Thread(target=rx_pdo_loop, name="rx_pdo_loop", daemon=True)
        tx_thread.start()
        rx_thread.start()

        while not rospy.is_shutdown():
            if not tx_thread.is_alive() or not rx_thread.is_alive():
                rospy.logerr("Worker thread exited unexpectedly, shutting down.")
                break
            time.sleep(0.1)
    
    finally:
        running.clear()
        time.sleep(0.05)
        cdpr.shut_down()
        
        
    
    
