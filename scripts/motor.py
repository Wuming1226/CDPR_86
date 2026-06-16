#! /usr/bin/env python

import argparse
import rospy
import os
import can
import time
from codec import int_to_le_bytes, le_bytes_to_int
import numpy as np


# CAN SDO Frame Data Format (Data: Little-endian format, LSB first)
# +----------+----------+----------+----------+----------+----------+----------+----------+
# |  Byte 0  |  Byte 1  |  Byte 2  |  Byte 3  |  Byte 4  |  Byte 5  |  Byte 6  |  Byte 7  |
# +----------+----------+----------+----------+----------+----------+----------+----------+
# |   CMD    |  Index L |  Index M | SubIndex |   Data   |   Data   |   Data   |   Data   |
# +----------+----------+----------+----------+----------+----------+----------+----------+

# CMD: write: request: 2F(1 byte) 2B(2 bytes) 27(3 bytes) 23(4 bytes)
#             answer:  60(success)  80(error)
#      read:  request: 40
#             answer:  4F(1 byte) 4B(2 bytes) 47(3 bytes) 43(4 bytes) 80(error)

class Motor:
    def __init__(self, can_channel, can_id, verbose=False):
        self.can_channel = can_channel
        self.can_bus = can.interface.Bus(channel=self.can_channel, bustype='socketcan', bitrate=500000)
        self.can_id = can_id
        self.req_address = 0x600 + can_id
        self.ans_address = 0x580 + can_id
        self.encoder_resolution = 10000
        self.ratio = 10
        self.verbose = verbose

        self.set_ratio(self.ratio)
        self.set_pdo()

    def __del__(self):
        self.can_bus.shutdown()
        
    def safe_send_can_msg(self, msg):
        try:
            self.can_bus.send(msg)
        except can.CanOperationError:
            self.shut_down()
            

    def set_ratio(self, ratio):
        """
        Set the transmission ratio: num of motor rotations / output rotations.
        """
        data = int_to_le_bytes(ratio)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x91, 0x60, 0x01, data[0], data[1], data[2], data[3]], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

    def set_pdo(self):
        # dump can bus buffer
        while True:
            msg = self.can_bus.recv(timeout=0.001)
            if msg is None:
                break
        # set PDO parameters
        # switch to pre-operation state
        msg = can.Message(arbitration_id=0x000, data=[0x80, 0x00 + self.can_id], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # deactivate PDO
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x00, 0x18, 0x01, 0x80 + self.can_id, 0x01, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x01, 0x18, 0x01, 0x80 + self.can_id, 0x02, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x02, 0x18, 0x01, 0x80 + self.can_id, 0x03, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x03, 0x18, 0x01, 0x80 + self.can_id, 0x04, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x00, 0x14, 0x01, 0x00 + self.can_id, 0x02, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x01, 0x14, 0x01, 0x00 + self.can_id, 0x03, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x02, 0x14, 0x01, 0x00 + self.can_id, 0x04, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x03, 0x14, 0x01, 0x00 + self.can_id, 0x05, 0x00, 0x80], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        if not self.verbose:
            # redefine PDO mappings
            # erase mapping (mapping object sub_index_0 -> 0)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x01, 0x1A, 0x00, 0x00, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            # set mapping object: motor position (user unit) 606400 32bit
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x01, 0x1A, 0x01, 0x20, 0x00, 0x63, 0x60], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            # set mapping object: motor position (user unit) 606400 32bit
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x01, 0x1A, 0x02, 0x20, 0x00, 0x64, 0x60], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            # set mapping number
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x01, 0x1A, 0x00, 0x02, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
        # set transmission type: cyclic synchronous 1
        if self.verbose:
            msg = can.Message(arbitration_id=self.req_address,
                              data=[0x2F, 0x00, 0x18, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x02, 0x18, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x03, 0x18, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x00, 0x14, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x01, 0x14, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x2F, 0x03, 0x14, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
        # 精简模式下仅设置TPDO2和RPDO3
        msg = can.Message(arbitration_id=self.req_address,
                        data=[0x2F, 0x01, 0x18, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2F, 0x02, 0x14, 0x02, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        # switch to operation state
        msg = can.Message(arbitration_id=0x000, data=[0x01, 0x00 + self.can_id], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # activate PDO
        if self.verbose:
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x00, 0x18, 0x01, 0x80 + self.can_id, 0x01, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x02, 0x18, 0x01, 0x80 + self.can_id, 0x03, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x03, 0x18, 0x01, 0x80 + self.can_id, 0x04, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x00, 0x14, 0x01, 0x00 + self.can_id, 0x02, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x01, 0x14, 0x01, 0x00 + self.can_id, 0x03, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address,
                            data=[0x23, 0x03, 0x14, 0x01, 0x00 + self.can_id, 0x05, 0x00, 0x00], is_extended_id=False)
            self.safe_send_can_msg(msg)
            self.show_sdo_response()
            time.sleep(0.002)
        # 精简模式下仅开启TPDO2和RPDO3
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x01, 0x18, 0x01, 0x80 + self.can_id, 0x02, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x02, 0x14, 0x01, 0x00 + self.can_id, 0x04, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        self.show_sdo_response()
        time.sleep(0.002)
        

    def enable_operation(self):
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x40, 0x60, 0x00, 0x0F, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

    def disable_operation(self):
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x40, 0x60, 0x00, 0x07, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

    def switch_on(self):
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x40, 0x60, 0x00, 0x07, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

    def shut_down(self):
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x40, 0x60, 0x00, 0x06, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

    def disable_voltage(self):
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x40, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        
    def show_sdo_response(self, time_out=0.5):
        start_time = time.time()
        while time.time() - start_time < time_out:
            msg = self.can_bus.recv(timeout=0.001)
            if msg is not None and msg.arbitration_id == self.ans_address:
                print([f'0x{b:02x}' for b in msg.data])
                break

    def set_profile_position_mode(self, target_position=0, velocity=1.0, relative=True):
        """
        Set the motor to profile position mode.
        """
        # disable
        self.disable_operation()

        # set running mode parameters
        # set to CiA402 mode
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x02, 0x20, 0x01, 0x00, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # set to profile position mode
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2F, 0x60, 0x60, 0x00, 0x01, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

        # set motion parameters
        # set target position
        target_position = round(target_position * self.encoder_resolution / (2 * np.pi))
        data = int_to_le_bytes(target_position)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x7A, 0x60, 0x00, data[0], data[1], data[2], data[3]], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # constant profile velocity
        velocity = round(velocity * self.encoder_resolution / (2 * np.pi))
        data = int_to_le_bytes(velocity)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x81, 0x60, 0x00, data[0], data[1], data[2], data[3]], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # profile acceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x83, 0x60, 0x00, 0x40, 0x9C, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # profile deceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x84, 0x60, 0x00, 0x40, 0x9C, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)

        # enable operation
        if relative:
            # control word 0x6F -> 0x7F
            msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x6F, 0x00, 0x00, 0x00],
                              is_extended_id=False)
            self.safe_send_can_msg(msg)
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x7F, 0x00, 0x00, 0x00],
                              is_extended_id=False)
            self.safe_send_can_msg(msg)
            time.sleep(0.002)
            print('MotorID: {}    Mode: Position Mode (Relative)    Position: {}'.format(
                self.can_id, target_position))

        else:
            # control word 0x2F -> 0x3F
            msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x2F, 0x00, 0x00, 0x00],
                              is_extended_id=False)
            self.safe_send_can_msg(msg)
            time.sleep(0.002)
            msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x3F, 0x00, 0x00, 0x00],
                              is_extended_id=False)
            self.safe_send_can_msg(msg)
            time.sleep(0.002)
            print('MotorID: {}    Mode: Position Mode (Absolute)    Position: {}'.format(
                self.can_id, target_position))
                
    def get_current_position(self, time_out=0.0075):
        msg = can.Message(arbitration_id=self.req_address, data=[0x40, 0x64, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        
        start_time = time.time()
        while time.time() - start_time < time_out:
            msg = self.can_bus.recv(timeout=0.001)
            if msg is not None and msg.arbitration_id == self.ans_address:
                if msg.data[1] == 0x64 and msg.data[2] == 0x60:
                    position = le_bytes_to_int(msg.data[4:8])
                    return position
            msg = can.Message(arbitration_id=self.req_address, data=[0x40, 0x64, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
                          is_extended_id=False)
            self.safe_send_can_msg(msg)
            time.sleep(0.002)

    def set_target_position(self, target_position=0):
        target_position = round(target_position * self.encoder_resolution / (2 * np.pi))
        data = int_to_le_bytes(target_position)
        msg = can.Message(arbitration_id=0x300 + self.can_id,
                          data=[data[0], data[1], data[2], data[3], 0x00, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)

    '''
    velocity: velocity of motor shaft (-3000 ~ 3000 rpm)
    '''
    def set_profile_velocity_mode(self, target_velocity=0):
        # disable
        self.disable_operation()
        # set to CiA402 mode
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2B, 0x02, 0x20, 0x01, 0x00, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # set to profile velocity mode
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x2F, 0x60, 0x60, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # max velocity
        msg = can.Message(arbitration_id=self.req_address, data=[0x23, 0x7F, 0x60, 0x00, 0x20, 0xA1, 0x07, 0x00],
                               is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # target velocity
        target_velocity = round(target_velocity * self.encoder_resolution / (2 * np.pi))
        data = int_to_le_bytes(target_velocity)
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0xFF, 0x60, 0x00, data[0], data[1], data[2], data[3]], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # acceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x83, 0x60, 0x00, 0xFF, 0xFF, 0xFF, 0xFF], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # deceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0x84, 0x60, 0x00, 0xFF, 0xFF, 0xFF, 0xFF], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # max acceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0xC5, 0x60, 0x00, 0xFF, 0xFF, 0xFF, 0xFF], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # max deceleration
        msg = can.Message(arbitration_id=self.req_address,
                          data=[0x23, 0xC6, 0x60, 0x00, 0xFF, 0xFF, 0xFF, 0xFF], is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # set to trapezoid slope
        msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x86, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        # shutdown
        msg2 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x06, 0x00, 0x00, 0x00],
                           is_extended_id=False)
        self.can_bus.send(msg2)
        time.sleep(0.002)
        # switch on
        msg3 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x07, 0x00, 0x00, 0x00],
                           is_extended_id=False)
        self.can_bus.send(msg3)
        time.sleep(0.002)
        # enable operation
        msg4 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x0F, 0x00, 0x00, 0x00],
                           is_extended_id=False)
        self.can_bus.send(msg4)
        time.sleep(0.002)

    def set_target_velocity(self, target_velocity=0):
        target_position = round(target_velocity * self.encoder_resolution / (2 * np.pi))
        data = int_to_le_bytes(target_position)
        msg = can.Message(arbitration_id=0x400 + self.can_id,
                          data=[0x00, 0x00, data[0], data[1], data[2], data[3], 0x00, 0x00], is_extended_id=False)
        self.safe_send_can_msg(msg)

        # info
        # print('MotoID: {}    Mode: Profile Velocity Mode    Velocity: {}'.format(self.canID, velocity))

    # def setTorMode(self, torque, velocity):
    #
    #     b, c, d, e = relist(torque * 10)
    #     # print(b,c,d,e)
    #
    #     j, k, m, n = relist(velocity)
    #     # print(j,k,m,n)
    #
    #     msg = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x02, 0x20, 0x01, 0x00, 0x00, 0x00, 0x00],
    #                       is_extended_id=False)
    #     self.safe_send_can_msg(msg)
    #
    #     Mode_Cho = can.Message(arbitration_id=self.req_address, data=[0x2F, 0x60, 0x60, 0x00, 0x04, 0x00, 0x00, 0x00],
    #                            is_extended_id=False)
    #     self.can_bus.send(Mode_Cho)
    #
    #     Tar_Tor = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x71, 0x60, 0x00, b, c, d, e],
    #                           is_extended_id=False)
    #     self.can_bus.send(Tar_Tor)
    #
    #     Pvelo_Max = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x07, 0x20, 0x10, j, k, m, n],
    #                             is_extended_id=False)
    #     self.can_bus.send(Pvelo_Max)
    #
    #     Nvelo_Max = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x07, 0x20, 0x11, j, k, m, n],
    #                             is_extended_id=False)
    #     self.can_bus.send(Nvelo_Max)
    #
    #     Type_Tor = can.Message(arbitration_id=self.req_address, data=[0x23, 0x88, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
    #                            is_extended_id=False)
    #     self.can_bus.send(Type_Tor)
    #
    #     Ramp_Tor = can.Message(arbitration_id=self.req_address, data=[0x23, 0x87, 0x60, 0x00, 0xFF, 0xFF, 0xFF, 0x0F],
    #                            is_extended_id=False)
    #     self.can_bus.send(Ramp_Tor)
    #
    #     msg2 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x06, 0x00, 0x00, 0x00],
    #                        is_extended_id=False)
    #     self.can_bus.send(msg2)
    #
    #     msg3 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x07, 0x00, 0x00, 0x00],
    #                        is_extended_id=False)
    #     self.can_bus.send(msg3)
    #
    #     msg4 = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x0F, 0x00, 0x00, 0x00],
    #                        is_extended_id=False)
    #     self.can_bus.send(msg4)

    def stop(self):
        stop = can.Message(arbitration_id=self.req_address, data=[0x2B, 0x40, 0x60, 0x00, 0x01, 0x00, 0x00, 0x00],
                           is_extended_id=False)
        self.can_bus.send(stop)

    # def setVelo(self, velocity):
    #
    #     self.can_bus = can.interface.Bus(channel=self.canChannel, bustype='socketcan')
    #
    #     b, c, d, e = relist(velocity * 10000 / 60)
    #     Tar_Velo = can.Message(arbitration_id=self.req_address, data=[0x23, 0xFF, 0x60, 0x00, b, c, d, e],
    #                            is_extended_id=False)
    #     self.can_bus.send(Tar_Velo)
    #     time.sleep(0.02)
    #
    #     # info
    #     # print('MotoID: {}    set Velocity: {}'.format(self.canID, velocity))
    #
    # def getVelo(self):
    #
    #     self.can_bus = can.interface.Bus(channel=self.canChannel, bustype='socketcan')
    #
    #     # request
    #     req = can.Message(arbitration_id=self.req_address, data=[0x40, 0x6C, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
    #                       is_extended_id=False)
    #
    #     while True:
    #         self.can_bus.send(req)
    #         time.sleep(0.002)
    #         ans = self.can_bus.recv()
    #
    #         if (ans.arbitration_id == self.ansAddress):
    #             data = decode(ans.data)
    #             break
    #
    #     return data
    #
    # def getPos(self):
    #
    #     self.can_bus = can.interface.Bus(channel=self.can_channel, bustype='socketcan')
    #
    #     req = can.Message(arbitration_id=self.req_address, data=[0x40, 0X64, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
    #                       is_extended_id=False)
    #
    #     while True:
    #         self.can_bus.send(req)
    #         time.sleep(0.002)
    #         ans = self.can_bus.recv()
    #
    #         if (ans.arbitration_id == self.ans_address):
    #             data = decode(ans.data)
    #             break
    #
    #     return data
    def get_sn_code(self, time_out=0.5):
        msg = can.Message(arbitration_id=self.req_address, data=[0x40, 0x18, 0x10, 0x04, 0x00, 0x00, 0x00, 0x00],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        
        start_time = time.time()
        while time.time() - start_time < time_out:
            msg = self.can_bus.recv(timeout=0.001)
            if msg is not None and msg.arbitration_id == self.ans_address:
                if msg.data[1] == 0x18 and msg.data[2] == 0x10:
                    sn_code = le_bytes_to_int(msg.data[4:8])
                    print("sn code: {}".format(sn_code))
                    print([f'0x{b:02x}' for b in msg.data])
                    return sn_code
                    
    def set_can_id(self, sn_code, can_id):
        sn_code_bytes = int_to_le_bytes(sn_code)
        msg = can.Message(arbitration_id=self.req_address, data=[0x23, 0x7D, 0x2F, 0x00, sn_code_bytes[0], sn_code_bytes[1], sn_code_bytes[2], sn_code_bytes[3]],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        self.show_sdo_response()
        
        msg = can.Message(arbitration_id=self.req_address, data=[0x23, 0x7E, 0x2F, 0x00, 0x00 + can_id, 0x00, 0x00, 0x00],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        self.show_sdo_response()
        
        msg = can.Message(arbitration_id=self.req_address, data=[0x23, 0x10, 0x10, 0x01, 0x73, 0x61, 0x76, 0x65],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        self.show_sdo_response()
        
        
    def get_error_code(self):
        msg = can.Message(arbitration_id=self.req_address, data=[0x40, 0x3F, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00],
                          is_extended_id=False)
        self.safe_send_can_msg(msg)
        time.sleep(0.002)
        self.show_sdo_response()
        


if __name__ == "__main__":
    # try:
    # rospy.init_node('motor', anonymous=True)
    # rate = rospy.Rate(20)

    _parser = argparse.ArgumentParser(
        description="Motor velocity test script: optional motor_id, run_time, velocity_cmd."
    )
    _parser.add_argument(
        "motor_id",
        nargs="?",
        type=int,
        default=5,
        help="CAN node id for Motor (default: 5)",
    )
    _parser.add_argument(
        "run_time",
        nargs="?",
        type=float,
        default=2.0,
        help="Velocity command duration in seconds (default: 2.0)",
    )
    _parser.add_argument(
        "velocity_cmd",
        nargs="?",
        type=float,
        default=-0.2,
        help="Velocity command sent by set_target_velocity (default: -0.2)",
    )
    _args = _parser.parse_args()
    motor_id = _args.motor_id
    run_time = _args.run_time
    velocity_cmd = _args.velocity_cmd

    os.system('sudo ip link set can0 type can bitrate 1000000')
    os.system('sudo ifconfig can0 up')

    bus = can.interface.Bus(channel='can0', bustype='socketcan')
    sync_msg = can.Message(arbitration_id=0x080, data=[0x01], is_extended_id=False)

    motor1 = Motor('can0', motor_id)

    # while True:
    #    print(motor1.getPos())
    #    rate.sleep()

    time.sleep(1)

    # motor1.set_profile_position_mode(1.57, relative=True)
    # motor1.switch_on()
    # motor1.disable_operation()
    # motor1.enable_operation()
    # motor1.set_profile_position_mode(target_position=0.*np.pi, velocity=0.5, relative=True)
    # motor1.set_profile_velocity_mode(0)
    
    # sn_code = motor1.get_sn_code()
    # motor1.set_can_id(sn_code, 6)
    motor1.get_error_code()


    start_time = time.time()
    print('start')
    # while time.time() - start_time < 3:
    #     motor1.set_target_velocity(np.pi)
    #     bus.send(sync_msg)
    #
    #     while True:
    #         ans = bus.recv(timeout=0.0001)
    #         if ans is not None and ans.arbitration_id == 0x281:
    #             data = le_bytes_to_int(ans.data[4:8])
    #             print(data)
    #             break
    #
    #     time.sleep(0.05)
    
    
    while time.time() - start_time < run_time:
        motor1.set_target_velocity(velocity_cmd)
        bus.send(sync_msg)

        cycle_start_time = time.time()
        while time.time() - cycle_start_time < 0.05:
            ans = bus.recv(timeout=0.0001)
            if ans is not None and ans.arbitration_id == 0x380 + motor_id:
                data = le_bytes_to_int(ans.data[0:4])
                print(data)
                break

        print("p: {}".format(motor1.get_current_position()))
        time.sleep(0.05)

    motor1.set_target_velocity(0)
    bus.send(sync_msg)
        
    # while time.time() - start_time < 0:
    #     motor1.set_target_velocity(0.5*np.pi)
    #     bus.send(sync_msg)

    #     cycle_start_time = time.time()
    #     while time.time() - cycle_start_time < 0.05:
    #         ans = bus.recv(timeout=0.0001)
    #         if ans is not None and ans.arbitration_id == 0x380 + motor_id:
    #             data = le_bytes_to_int(ans.data[0:4])
    #             print(data)
    #             break

    #     time.sleep(0.05)
        
    # time.sleep(5)

    # print(motor1.getPos())
    #motor1.disable_operation()
    # time.sleep(2)
    # motor1.shut_down()
    # time.sleep(5)
    # motor1.setVelo(100)
    # time.sleep(1.5)
    # motor1.setProfVeloMode(0)

#        cnt = 0
#        while cnt < 20:
#            print("Velocity: {}".format(motor1.getVelo()))
#            rate.sleep()
#            cnt += 1
#        motor1.stop()

# except rospy.ROSInterruptException:
#     motor1.stop()



