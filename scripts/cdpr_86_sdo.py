#! /usr/bin/env python3

import os
import rospy
import time
import numpy as np
import can

from motor import Motor

from std_msgs.msg import Int16MultiArray, Int64MultiArray, Float32MultiArray
#from slave.srv import SetMotorVelo, SetMotorVeloResponse, GetMotorPos, GetMotorPosResponse


class CDPR:
    def __init__(self, motor1_id=1, motor2_id=2, motor3_id=3, motor4_id=4, motor5_id=5, motor6_id=6, motor7_id=7, motor8_id=8):

        # can settings
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 up')
        os.system('sudo ifconfig can0 txqueuelen 10000')
        self.can_channel = 'can0'
        
        # ros settings
        rospy.init_node('cdpr_86_actuator', anonymous=False)
        rospy.Subscriber('motor_velo', Float32MultiArray, callback=self.velo_callback)
        self.motor_pos_pub = rospy.Publisher('motor_pos', Float32MultiArray, queue_size=10)
        
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
        
        self.motor_pos1 = 0
        self.motor_pos2 = 0
        self.motor_pos3 = 0
        self.motor_pos4 = 0
        self.motor_pos5 = 0
        self.motor_pos6 = 0
        self.motor_pos7 = 0
        self.motor_pos8 = 0
        self.init_motor_pos = np.zeros(8, dtype=np.float64)
        
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
        print("target velocity: {}".format(msg.data))
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

    control_hz = 15.0
    sensor_hz = 30.0
    control_period = 1.0 / control_hz
    sensor_period = 1.0 / sensor_hz
    #cnt = 0
    #while not rospy.is_shutdown():
    #    #cdpr.receiveMotorVelo()
    #    cdpr.pub_motor_pos()
    #    rate.sleep()
    
    bus = can.interface.Bus(channel='can0', bustype='socketcan')
    sync_msg = can.Message(arbitration_id=0x080, data=[0x01], is_extended_id=False)

    # Initialize cached motor positions: block until all eight SDO reads succeed.
    while not rospy.is_shutdown():
        pos1 = cdpr._motor1.get_current_position()
        pos2 = cdpr._motor2.get_current_position()
        pos3 = cdpr._motor3.get_current_position()
        pos4 = cdpr._motor4.get_current_position()
        pos5 = cdpr._motor5.get_current_position()
        pos6 = cdpr._motor6.get_current_position()
        pos7 = cdpr._motor7.get_current_position()
        pos8 = cdpr._motor8.get_current_position()

        if None not in [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]:
            cdpr.motor_pos1 = pos1
            cdpr.motor_pos2 = pos2
            cdpr.motor_pos3 = pos3
            cdpr.motor_pos4 = pos4
            cdpr.motor_pos5 = pos5
            cdpr.motor_pos6 = pos6
            cdpr.motor_pos7 = pos7
            cdpr.motor_pos8 = pos8
            cdpr.init_motor_pos = np.array([
                cdpr.motor_pos1, cdpr.motor_pos2, cdpr.motor_pos3, cdpr.motor_pos4,
                cdpr.motor_pos5, cdpr.motor_pos6, cdpr.motor_pos7, cdpr.motor_pos8
            ], dtype=np.float64)
            break
            
        missing_initial_motors = [
            f"motor{i + 1}" for i, pos in enumerate([pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]) if pos is None
        ]
        rospy.logwarn("Waiting initial motor positions, read None from: {}".format(", ".join(missing_initial_motors)))
        time.sleep(0.01)
    
    try:
        next_control_time = time.monotonic()
        next_sensor_time = time.monotonic()

        while not rospy.is_shutdown():
            now = time.monotonic()

            if now >= next_control_time:
                bus.send(sync_msg)
                next_control_time += control_period

                if time.time() - cdpr.last_velo_callback_time > cdpr.max_interval:  # 规定时间间隔内没接收到速度指令则停机
                    cdpr.set_velocity([0, 0, 0, 0, 0, 0, 0, 0])
                    print('No signal')

            #if cdpr.exceed_cnt >= cdpr.exceed_tol:  # 连续接收最大速度指令，则判断为发生错误，停机
            #    cdpr.motor1.set_position(cdpr.motor1.get_position())     # 锁定电机在当前位置
            #    cdpr.motor2.set_position(cdpr.motor2.get_position())
            #    cdpr.motor3.set_position(cdpr.motor3.get_position())
            #    cdpr.motor4.set_position(cdpr.motor4.get_position())
            #    print('Over speed')

            if now >= next_sensor_time:
                pos1 = cdpr._motor1.get_current_position()
                pos2 = cdpr._motor2.get_current_position()
                pos3 = cdpr._motor3.get_current_position()
                pos4 = cdpr._motor4.get_current_position()
                pos5 = cdpr._motor5.get_current_position()
                pos6 = cdpr._motor6.get_current_position()
                pos7 = cdpr._motor7.get_current_position()
                pos8 = cdpr._motor8.get_current_position()

                if pos1 is not None:
                    cdpr.motor_pos1 = pos1
                if pos2 is not None:
                    cdpr.motor_pos2 = pos2
                if pos3 is not None:
                    cdpr.motor_pos3 = pos3
                if pos4 is not None:
                    cdpr.motor_pos4 = pos4
                if pos5 is not None:
                    cdpr.motor_pos5 = pos5
                if pos6 is not None:
                    cdpr.motor_pos6 = pos6
                if pos7 is not None:
                    cdpr.motor_pos7 = pos7
                if pos8 is not None:
                    cdpr.motor_pos8 = pos8

                missing_runtime_motors = [
                    f"motor{i + 1}" for i, pos in enumerate([pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]) if pos is None
                ]
                if missing_runtime_motors:
                    rospy.logwarn("Motor position read None from: {}".format(", ".join(missing_runtime_motors)))

                cdpr.motor_pos_pub.publish(Float32MultiArray(
                                           data=(np.array([
                                               cdpr.motor_pos1, cdpr.motor_pos2, cdpr.motor_pos3, cdpr.motor_pos4,
                                               cdpr.motor_pos5, cdpr.motor_pos6, cdpr.motor_pos7, cdpr.motor_pos8
                                           ], dtype=np.float64) - cdpr.init_motor_pos)
                                           / cdpr._motor1.encoder_resolution * 2 * np.pi))
                next_sensor_time += sensor_period

            # keep loop responsive while preserving the two target rates
            time.sleep(0.001)
            
        

        cdpr.set_velocity([0, 0, 0, 0, 0, 0, 0, 0])
        cdpr.shut_down()
    
    finally:
        cdpr.shut_down()
        
        
    
    
