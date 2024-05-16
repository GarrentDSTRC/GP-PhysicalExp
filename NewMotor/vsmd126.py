#coding:utf-8
from ctypes import *
import platform
from time import sleep
from usb_device import *
from usb2can import *
from time import perf_counter
import numpy as np

class easy_can():
    def __init__(self):
        self.CAN1 = 1
        self.DevHandles = (c_uint * 20)()
        # 扫描设备并将设备号存放到设备号数组中
        ret = USB_ScanDevice(byref(self.DevHandles))
        if (ret == 0):
            print("No device connected!")
            exit()
        else:
            print("Have %d device connected!" % ret)
        # 打开设备
        ret = USB_OpenDevice(self.DevHandles[0])
        if (bool(ret)):
            print("Open device success!")
        else:
            print("Open device faild!")
            exit()
        # 初始化CAN
        CANConfig = CAN_INIT_CONFIG()

        # 获取波特率参数
        ret = CAN_GetCANSpeedArg(self.DevHandles[0], byref(CANConfig), 1000000)
        if (ret != CAN_SUCCESS):
            print("Get CAN speed failed!")
            exit()
        else:
            print("Get CAN speed Success!")
        ret = CAN_Init(self.DevHandles[0], self.CAN1, byref(CANConfig))
        if (ret != CAN_SUCCESS):
            print("Config CAN failed!")
            exit()
        else:
            print("Config CAN Success!")
        self.CanMsg = (CAN_MSG * 3)()


    def close(self):
            # Close device
            ret = USB_CloseDevice(self.DevHandles[0])
            if (bool(ret)):
                print("Close device success!")
            else:
                print("Close device faild!")
                exit()

    def getValue(self,value):
        a0 = (value & 0xff000000) >> 24
        a1 = (value & 0x00ff0000) >> 16
        a2 = (value & 0x0000ff00) >> 8
        a3 = (value & 0x000000ff)
        return [a0,a1,a2,a3]

    def vsmd126_enable(self):
        self.CanMsg[1].ExternFlag = 0
        self.CanMsg[1].RemoteFlag = 0
        self.CanMsg[1].ID = 0x00E1
        self.CanMsg[1].DataLen = 0
        # self.CanMsg[1].Data[0] = 0
        # self.CanMsg[1].Data[1] = 0
        # self.CanMsg[1].Data[2] = 0
        # self.CanMsg[1].Data[3] = 0
        # self.CanMsg[1].Data[4] = 0
        # self.CanMsg[1].Data[5] = 0
        # self.CanMsg[1].Data[6] = 0
        # self.CanMsg[1].Data[7] = 0
        SendedNum = CAN_SendMsg(self.DevHandles[0], self.CAN1, byref(self.CanMsg[1]), 1)
        if SendedNum >= 0:
            print("vsmd126 enable!" )
        else:
            print("vsmd126 enable failed!")

    def vsmd126_abs_loc(self,angle):
        # 发送CAN帧
        value = int(angle)
        value_ = self.getValue(value)
        self.CanMsg[0].ExternFlag = 0
        self.CanMsg[0].RemoteFlag = 0
        self.CanMsg[0].ID = 0x00E6
        self.CanMsg[0].DataLen = 4
        self.CanMsg[0].Data[0] = value_[0]
        self.CanMsg[0].Data[1] = value_[1]
        self.CanMsg[0].Data[2] = value_[2]
        self.CanMsg[0].Data[3] = value_[3]
        # self.CanMsg[0].Data[4] = value_[0]
        # self.CanMsg[0].Data[5] = value_[1]
        # self.CanMsg[0].Data[6] = value_[2]
        # self.CanMsg[0].Data[7] = value_[3]
        SendedNum = CAN_SendMsg(self.DevHandles[0], self.CAN1, byref(self.CanMsg[0]), 1)
        # if SendedNum >= 0:
        #     print("Success send frames:%d" % SendedNum)
        # else:
        #     print("Send CAN data failed!")
        # Delay
        # CanNum = 0
        # while(CanNum<=0):
        #     self.CanMsgBuffer = (CAN_MSG * 10240)()
        #     CanNum = CAN_GetMsg(self.DevHandles[0], self.CAN1, byref(self.CanMsgBuffer))
        #
        # if CanNum > 0:
        #     print("CanNum = %d" % CanNum)
        #     for i in range(0, CanNum):
        #         print("CanMsg[%d].ID = %d" % (i, self.CanMsgBuffer[i].ID))
        #         print("CanMsg[%d].TimeStamp = %d" % (i, self.CanMsgBuffer[i].TimeStamp))
        #         print("CanMsg[%d].Data = " % i, end='')
        #         for j in range(0, self.CanMsgBuffer[i].DataLen):
        #             print("%02X " % self.CanMsgBuffer[i].Data[j], end='')
        #         print("")
        # elif CanNum == 0:
        #     print("No CAN data!")
        # else:
        #     print("Get CAN data error!")


if __name__ == '__main__':
    can = easy_can()
    now = perf_counter()
    last = perf_counter()
    can.vsmd126_enable()
    sleep(0.5)
    while True:
        can.vsmd126_abs_loc(6400*90/360*np.sin(2*np.pi/2*(perf_counter()-now)))
        print(perf_counter())
    # can.close()




