#coding:utf-8
from ctypes import *
import platform
from time import sleep
from usb_device import *
from usb2can import *


if __name__ == '__main__': 
    CAN1 = 0
    CAN2 = 1
    DevHandles = (c_uint * 20)()
    # 扫描设备并将设备号存放到设备号数组中
    ret = USB_ScanDevice(byref(DevHandles))
    if(ret == 0):
        print("No device connected!")
        exit()
    else:
        print("Have %d device connected!"%ret)
    # 打开设备
    ret = USB_OpenDevice(DevHandles[0])
    if(bool(ret)):
        print("Open device success!")
    else:
        print("Open device faild!")
        exit()
    # 获取设备固件信息
    USB2XXXInfo = DEVICE_INFO()
    USB2XXXFunctionString = (c_char * 256)()
    ret = DEV_GetDeviceInfo(DevHandles[0],byref(USB2XXXInfo),byref(USB2XXXFunctionString))
    if(bool(ret)):
        print("USB2XXX device infomation:")
        print("--Firmware Name: %s"%bytes(USB2XXXInfo.FirmwareName).decode('ascii'))
        print("--Firmware Version: v%d.%d.%d"%((USB2XXXInfo.FirmwareVersion>>24)&0xFF,(USB2XXXInfo.FirmwareVersion>>16)&0xFF,USB2XXXInfo.FirmwareVersion&0xFFFF))
        print("--Hardware Version: v%d.%d.%d"%((USB2XXXInfo.HardwareVersion>>24)&0xFF,(USB2XXXInfo.HardwareVersion>>16)&0xFF,USB2XXXInfo.HardwareVersion&0xFFFF))
        print("--Build Date: %s"%bytes(USB2XXXInfo.BuildDate).decode('ascii'))
        print("--Serial Number: ",end='')
        for i in range(0, len(USB2XXXInfo.SerialNumber)):
            print("%08X"%USB2XXXInfo.SerialNumber[i],end='')
        print("")
        print("--Function String: %s"%bytes(USB2XXXFunctionString.value).decode('ascii'))
    else:
        print("Get device infomation faild!")
        exit()
    # 初始化CAN
    CANConfig = CAN_INIT_CONFIG()
    ret = CAN_GetCANSpeedArg(DevHandles[0],byref(CANConfig),500000)
    if(ret != CAN_SUCCESS):
        print("Get CAN speed failed!")
        exit()
    else:
        print("Get CAN speed Success!")
    ret = CAN_Init(DevHandles[0],CAN1,byref(CANConfig))
    if(ret != CAN_SUCCESS):
        print("Config CAN1 failed!")
        exit()
    else:
        print("Config CAN1 Success!")
    ret = CAN_Init(DevHandles[0],CAN2,byref(CANConfig))
    if(ret != CAN_SUCCESS):
        print("Config CAN2 failed!")
        exit()
    else:
        print("Config CAN2 Success!")
    # 配置过滤器，可以不配置，默认接收所有数据
    CANFilter = CAN_FILTER_CONFIG()
    CANFilter.Enable = 1
    CANFilter.ExtFrame = 0
    CANFilter.FilterIndex = 0
    CANFilter.FilterMode = 0
    CANFilter.MASK_IDE = 0
    CANFilter.MASK_RTR = 0
    CANFilter.MASK_Std_Ext = 0
    ret = CAN_Filter_Init(DevHandles[0],CAN1,byref(CANFilter))
    if(ret != CAN_SUCCESS):
        print("Config CAN1 Filter failed!")
        exit()
    else:
        print("Config CAN1 Filter Success!")
    ret = CAN_Filter_Init(DevHandles[0],CAN2,byref(CANFilter))
    if(ret != CAN_SUCCESS):
        print("Config CAN2 Filter failed!")
        exit()
    else:
        print("Config CAN2 Filter Success!")
    # 发送CAN帧
    CanMsg = (CAN_MSG*5)()
    for i in range(0,5):
        CanMsg[i].ExternFlag = 0
        CanMsg[i].RemoteFlag = 0
        CanMsg[i].ID = i
        CanMsg[i].DataLen = 8
        for j in range(0,CanMsg[i].DataLen):
            CanMsg[i].Data[j] = (i<<4)|j
    SendedNum = CAN_SendMsg(DevHandles[0],CAN1,byref(CanMsg),5)
    if SendedNum >= 0 :
        print("CAN1 success send frames:%d"%SendedNum)
    else:
        print("Send CAN1 data failed!")
    # Delay
    sleep(0.5)
    # 读取CAN数据
    CanMsgBuffer = (CAN_MSG*10240)()
    CanNum = CAN_GetMsg(DevHandles[0],CAN2,byref(CanMsgBuffer))
    if CanNum > 0:
        print("CAN2 CanNum = %d"%CanNum)
        for i in range(0,CanNum):
            print("CanMsg[%d].ID = %d"%(i,CanMsgBuffer[i].ID))
            print("CanMsg[%d].TimeStamp = %d"%(i,CanMsgBuffer[i].TimeStamp))
            print("CanMsg[%d].Data = "%i,end='')
            for j in range(0,CanMsgBuffer[i].DataLen):
                print("%02X "%CanMsgBuffer[i].Data[j],end='')
            print("")
    elif CanNum == 0:
        print("No CAN2 data!")
    else:
        print("Get CAN2 data error!")

    # 发送CAN帧
    CanMsg = (CAN_MSG*5)()
    for i in range(0,5):
        CanMsg[i].ExternFlag = 0
        CanMsg[i].RemoteFlag = 0
        CanMsg[i].ID = i
        CanMsg[i].DataLen = 8
        for j in range(0,CanMsg[i].DataLen):
            CanMsg[i].Data[j] = (i<<4)|j
    SendedNum = CAN_SendMsg(DevHandles[0],CAN2,byref(CanMsg),5)
    if SendedNum >= 0 :
        print("CAN2 success send frames:%d"%SendedNum)
    else:
        print("Send CAN2 data failed!")
    # Delay
    sleep(0.5)
    # 读取CAN数据
    CanMsgBuffer = (CAN_MSG*10240)()
    CanNum = CAN_GetMsg(DevHandles[0],CAN1,byref(CanMsgBuffer))
    if CanNum > 0:
        print("CAN2 CanNum = %d"%CanNum)
        for i in range(0,CanNum):
            print("CanMsg[%d].ID = %d"%(i,CanMsgBuffer[i].ID))
            print("CanMsg[%d].TimeStamp = %d"%(i,CanMsgBuffer[i].TimeStamp))
            print("CanMsg[%d].Data = "%i,end='')
            for j in range(0,CanMsgBuffer[i].DataLen):
                print("%02X "%CanMsgBuffer[i].Data[j],end='')
            print("")
    elif CanNum == 0:
        print("No CAN1 data!")
    else:
        print("Get CAN1 data error!")
    # Close device
    ret = USB_CloseDevice(DevHandles[0])
    if(bool(ret)):
        print("Close device success!")
    else:
        print("Close device faild!")
        exit()
