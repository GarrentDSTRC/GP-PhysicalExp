#coding:utf-8
from NewMotor.usb2can import *
import multiprocessing

class CANdev(multiprocessing.Process):
    def __init__(self,dict):
        super(CANdev, self).__init__()
        self.daemon = True
        self.dict = dict
        self.CAN1 = 1

    def init(self):
        self.DevHandles = (c_uint * 20)()
        self.dict['angle1'] = 0.0
        self.dict['angle2'] = 0.0
        self.dict['angle3'] = 0.0
        self.dict['angle4'] = 0.0
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
        self.CanMsg = (CAN_MSG * 4)()
        for i in range(4):
            self.CanMsg[i].ExternFlag = 0
            self.CanMsg[i].RemoteFlag = 0
            self.CanMsg[i].ID = 321 + i
            self.CanMsg[i].DataLen = 8
            self.CanMsg[i].Data[0] = 163
            self.CanMsg[i].Data[1] = 0
            self.CanMsg[i].Data[2] = 0
            self.CanMsg[i].Data[3] = 0


    def close(self):
            # Close device
            ret = USB_CloseDevice(self.DevHandles[0])
            if (bool(ret)):
                print("Close device success!")
            else:
                print("Close device faild!")
                exit()

    def getValue(self,value):
        a3 = (value & 0xff000000) >> 24
        a2 = (value & 0x00ff0000) >> 16
        a1 = (value & 0x0000ff00) >> 8
        a0 = (value & 0x000000ff)
        return [a0,a1,a2,a3]

    def run(self):
        self.init()
        while True:
        # 发送CAN帧
            value1 = int(self.dict['heave_1'] * 100)
            value2 = int(self.dict['pitch_2'] * 100)
            value3 = int(self.dict['heave_2'] * 100)
            value4 = int(self.dict['pitch_1'] * 100)
            # print(f'value1:{value1},value2:{value2},value3:{value3},value4:{value4}')
            value1_ = self.getValue(value1)
            value2_ = self.getValue(value2)
            value3_ = self.getValue(value3)
            value4_ = self.getValue(value4)
            self.CanMsg[0].Data[4] = value1_[0]
            self.CanMsg[0].Data[5] = value1_[1]
            self.CanMsg[0].Data[6] = value1_[2]
            self.CanMsg[0].Data[7] = value1_[3]
            self.CanMsg[1].Data[4] = value2_[0]
            self.CanMsg[1].Data[5] = value2_[1]
            self.CanMsg[1].Data[6] = value2_[2]
            self.CanMsg[1].Data[7] = value2_[3]
            self.CanMsg[2].Data[4] = value3_[0]
            self.CanMsg[2].Data[5] = value3_[1]
            self.CanMsg[2].Data[6] = value3_[2]
            self.CanMsg[2].Data[7] = value3_[3]
            self.CanMsg[3].Data[4] = value4_[0]
            self.CanMsg[3].Data[5] = value4_[1]
            self.CanMsg[3].Data[6] = value4_[2]
            self.CanMsg[3].Data[7] = value4_[3]
            CAN_SendMsg(self.DevHandles[0], self.CAN1, byref(self.CanMsg), 4)
            # if SendedNum >= 0:
            #     print("Success send frames:%d" % SendedNum)
            # else:
            #     print("Send CAN data failed!")
            # Delay
            CanNum = 0
            while(CanNum<=0):
                self.CanMsgBuffer = (CAN_MSG * 10240)()
                CanNum = CAN_GetMsg(self.DevHandles[0], self.CAN1, byref(self.CanMsgBuffer))







