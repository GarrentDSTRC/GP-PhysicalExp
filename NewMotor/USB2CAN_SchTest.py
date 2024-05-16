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
    # 获取波特率参数
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
    # 调度表方式发送CAN帧
    CanMsg = (CAN_MSG*10)()
    for i in range(0,10):
        CanMsg[i].ExternFlag = 0
        CanMsg[i].RemoteFlag = 0
        CanMsg[i].ID = i
        CanMsg[i].DataLen = 8
        CanMsg[i].TimeStamp = 100
        for j in range(0,CanMsg[i].DataLen):
            CanMsg[i].Data[j] = (i<<4)|j
    MsgNum = (c_ubyte * 3)()    #配置3个调度表
    MsgNum[0] = 2   #第一个调度表里面有2帧数据
    MsgNum[1] = 5   #第二个调度表里面有5帧数据
    MsgNum[2] = 3   #第三个调度表里面有3帧数据
    SendTimes = (c_uint16 * 3)() #配置每个调度表里面的帧循环发送次数，0xFFFF为一直发送
    SendTimes[0] = 5        #第一个表里面的2帧循环发送5次
    SendTimes[1] = 10       #第二个表里面的5帧循环发送10次
    SendTimes[2] = 0xFFFF   #第三个表里面的3帧一直循环发送
    ret = CAN_SetSchedule(DevHandles[0],CAN1, byref(CanMsg),MsgNum,SendTimes,3)
    if(ret != CAN_SUCCESS):
        print("Set Schedule CAN1 Failed!")
        print("ret = %d"%ret)
        exit()
    else:
        print("Set Schedule CAN1 Success!")
    #启动第一个调度表，时间精度设置为10ms，调度表里面的帧顺序发送
    #注意：同一时间只能执行一个调度表，若没有停止已经启动的调度表，再次调用启动调度表函数，将会自动停止之前已经启动了的调度表
    ret = CAN_StartSchedule(DevHandles[0],CAN1,0,10,1)
    if(ret != CAN_SUCCESS):
        print("Start Schedule CAN1 Failed!")
        exit()
    else:
        print("Start Schedule CAN1 Success!")
    sleep(1)
    # 循环读取CAN数据
    CanMsgBuffer = (CAN_MSG*10240)()
    ReadMsgTimes = 100
    while(ReadMsgTimes > 0):
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
        # Delay
        sleep(0.1)
        ReadMsgTimes = ReadMsgTimes-1
    #停止调度表
    CAN_StopSchedule(DevHandles[0],CAN1)
    exit()
