#! /usr/bin/env python3

import os
import rospy
import time
import numpy as np
import queue
import can

from motor_pos import Motor
from codec import le_bytes_to_int

from std_msgs.msg import Int16MultiArray, Int64MultiArray, Float32MultiArray
#from slave.srv import SetMotorVelo, SetMotorVeloResponse, GetMotorPos, GetMotorPosResponse


class CDPR:
    def __init__(self, motor1_id=1, motor2_id=2, motor3_id=3, motor4_id=4, motor5_id=5, motor6_id=6, motor7_id=7, motor8_id=8):

        # can settings
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 up')
        os.system('sudo ifconfig can0 txqueuelen 1000')
        self.can_channel = 'can0'
        
        # ros settings
        rospy.init_node('cdpr_86', anonymous=True)
        #self._setMotorVeloSrv = rospy.Service('set_motor_velo', SetMotorVelo, self._setMotorVeloHandle)
        #self._getMotorPosSrv = rospy.Service('get_motor_pos', GetMotorPos, self._getMotorPosHandle)
        rospy.Subscriber('motor_velo', Float32MultiArray, callback=self.velo_callback)
        rospy.Subscriber('motor_pos_com', Float32MultiArray, callback=self.pos_com_callback)
        #self._motor_pos_pub = rospy.Publisher('motor_pos', Int64MultiArray, queue_size=10)
        
        # safety settings
        self.last_velo_callback_time = time.time()
        self.last_pos_callback_time = time.time()
        self.max_interval = 0.5

        # initialize motors
        run_velocity = 2
        self._motor1 = Motor(self.can_channel, motor1_id)
        self._motor1.set_profile_position_mode(velocity=run_velocity)
        self._motor2 = Motor(self.can_channel, motor2_id)
        self._motor2.set_profile_position_mode(velocity=run_velocity)
        self._motor3 = Motor(self.can_channel, motor3_id)
        self._motor3.set_profile_position_mode(velocity=run_velocity)
        self._motor4 = Motor(self.can_channel, motor4_id)
        self._motor4.set_profile_position_mode(velocity=run_velocity)
        self._motor5 = Motor(self.can_channel, motor5_id)
        self._motor5.set_profile_position_mode(velocity=run_velocity)
        self._motor6 = Motor(self.can_channel, motor6_id)
        self._motor6.set_profile_position_mode(velocity=run_velocity)
        self._motor7 = Motor(self.can_channel, motor7_id)
        self._motor7.set_profile_position_mode(velocity=run_velocity)
        self._motor8 = Motor(self.can_channel, motor8_id)
        self._motor8.set_profile_position_mode(velocity=run_velocity)
        
        time.sleep(1)
        init_motor_pos1 = self._motor1.get_current_position()
        init_motor_pos2 = self._motor2.get_current_position()
        init_motor_pos3 = self._motor3.get_current_position()
        init_motor_pos4 = self._motor4.get_current_position()
        init_motor_pos5 = self._motor5.get_current_position()
        init_motor_pos6 = self._motor6.get_current_position()
        init_motor_pos7 = self._motor7.get_current_position()
        init_motor_pos8 = self._motor8.get_current_position()
        self.init_motor_pos = np.array([init_motor_pos1,
                                        init_motor_pos2,
                                        init_motor_pos3,
                                        init_motor_pos4,
                                        init_motor_pos5,
                                        init_motor_pos6,
                                        init_motor_pos7,
                                        init_motor_pos8])
        
        print("init_motor_pos: {}".format(self.init_motor_pos))
        
    def set_position(self, pos):
        self._motor1.set_target_position_abs(pos[0] + self.init_motor_pos[0] / 10000 * 2 * np.pi)
        self._motor2.set_target_position_abs(pos[1] + self.init_motor_pos[1] / 10000 * 2 * np.pi)
        self._motor3.set_target_position_abs(pos[2] + self.init_motor_pos[2] / 10000 * 2 * np.pi)
        self._motor4.set_target_position_abs(pos[3] + self.init_motor_pos[3] / 10000 * 2 * np.pi)
        self._motor5.set_target_position_abs(pos[4] + self.init_motor_pos[4] / 10000 * 2 * np.pi)
        self._motor6.set_target_position_abs(pos[5] + self.init_motor_pos[5] / 10000 * 2 * np.pi)
        self._motor7.set_target_position_abs(pos[6] + self.init_motor_pos[6] / 10000 * 2 * np.pi)
        self._motor8.set_target_position_abs(pos[7] + self.init_motor_pos[7] / 10000 * 2 * np.pi)
        
    def stop(self):
        self._motor1.set_target_position_rel(0)
        self._motor2.set_target_position_rel(0)
        self._motor3.set_target_position_rel(0)
        self._motor4.set_target_position_rel(0)
        self._motor5.set_target_position_rel(0)
        self._motor6.set_target_position_rel(0)
        self._motor7.set_target_position_rel(0)
        self._motor8.set_target_position_rel(0)
        
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
        
    def pos_com_callback(self, msg):
        print("target position: {}".format(msg.data))
        self.last_pos_callback_time = time.time()
        self.set_position(msg.data)
        
    # def pub_motor_pos(self):
    #     motor_pos = Int64MultiArray(data=np.array([self._motor1.get_current_position(), self._motor2.get_current_position()
    #                                                self._motor3.get_current_position(), self._motor4.get_current_position()
    #                                                self._motor5.get_current_position(), self._motor6.get_current_position()
    #                                                self._motor7.get_current_position(), self._motor8.get_current_position()]))
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

    rate = rospy.Rate(20)
    #cnt = 0
    #while not rospy.is_shutdown():
    #    #cdpr.receiveMotorVelo()
    #    cdpr.pub_motor_pos()
    #    rate.sleep()
    
    bus = can.interface.Bus(channel='can0', bustype='socketcan')
    sync_msg = can.Message(arbitration_id=0x080, data=[0x01], is_extended_id=False)
    
    
    while not rospy.is_shutdown():
    
        # bus.send(sync_msg)     

        if time.time() - cdpr.last_pos_callback_time > cdpr.max_interval:  # 规定时间间隔内没接收到速度指令则停机
            cdpr.stop()
            print('No signal')

        #if cdpr.exceed_cnt >= cdpr.exceed_tol:  # 连续接收最大速度指令，则判断为发生错误，停机
        #    cdpr.motor1.set_position(cdpr.motor1.get_position())     # 锁定电机在当前位置
        #    cdpr.motor2.set_position(cdpr.motor2.get_position())
        #    cdpr.motor3.set_position(cdpr.motor3.get_position())
        #    cdpr.motor4.set_position(cdpr.motor4.get_position())
        #    print('Over speed')

        # cdpr.get_and_pub_motor_pos()
        rate.sleep()

    #cdpr.set_velocity([0, 0, 0, 0, 0, 0, 0, 0])
    init_motor_pos1 = cdpr._motor1.get_current_position()
    init_motor_pos2 = cdpr._motor2.get_current_position()
    init_motor_pos3 = cdpr._motor3.get_current_position()
    init_motor_pos4 = cdpr._motor4.get_current_position()
    init_motor_pos5 = cdpr._motor5.get_current_position()
    init_motor_pos6 = cdpr._motor6.get_current_position()
    init_motor_pos7 = cdpr._motor7.get_current_position()
    init_motor_pos8 = cdpr._motor8.get_current_position()
    print(np.array([init_motor_pos1,
                    init_motor_pos2,
                    init_motor_pos3,
                    init_motor_pos4,
                    init_motor_pos5,
                    init_motor_pos6,
                    init_motor_pos7,
                    init_motor_pos8]))
    cdpr.shut_down()
    
    
