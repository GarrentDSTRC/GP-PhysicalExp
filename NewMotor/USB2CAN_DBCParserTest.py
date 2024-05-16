#coding:utf-8
from ctypes import *
import platform
from time import sleep
from usb_device import *
from usb2can import *
from dbc_parser import *

if __name__ == '__main__': 
    libBuildData = (c_char * 32)()
    # 获取库编译日期
    DEV_GetDllBuildTime(byref(libBuildData))
    print("lib build date time: %s"%bytes(libBuildData).decode('ascii'))
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
    # 解析DBC文件
    DBCHandle = DBC_ParserFile(DevHandles[0],b"demo.dbc")
    if(DBCHandle==0):
        print("Parser DBC File error!")
        exit()
    else:
        print("Parser DBC File success!")
    # 打印DBC里面报文和信号相关信息
    DBCMsgNum = DBC_GetMsgQuantity(c_uint64(DBCHandle))
    for i in range(0,DBCMsgNum):
        MsgName = (c_char * 64)()
        DBC_GetMsgName(c_uint64(DBCHandle),i,MsgName)
        print("Msg.Name = %s"%bytes(MsgName.value).decode('ascii'))
        DBCSigNum = DBC_GetMsgSignalQuantity(c_uint64(DBCHandle),MsgName)
        print("Signals:")
        for j in range(0,DBCSigNum):
            SigName = (c_char * 64)()
            DBC_GetMsgSignalName(c_uint64(DBCHandle),MsgName,j,SigName)
            print("\t%s "%bytes(SigName.value).decode('ascii'))
        print("")
    # 设置信号值
    DBC_SetSignalValue(c_uint64(DBCHandle),b"msg_moto_speed",b"moto_speed",c_double(2412))
    DBC_SetSignalValue(c_uint64(DBCHandle),b"msg_oil_pressure",b"oil_pressure",c_double(980))
    DBC_SetSignalValue(c_uint64(DBCHandle),b"msg_speed_can",b"speed_can",c_double(120))
    # 将信号值填入CAN消息里面
    CanMsg = (CAN_MSG*5)()
    DBC_SyncValueToCANMsg(c_uint64(DBCHandle),b"msg_moto_speed",byref(CanMsg[0]))
    DBC_SyncValueToCANMsg(c_uint64(DBCHandle),b"msg_oil_pressure",byref(CanMsg[1]))
    DBC_SyncValueToCANMsg(c_uint64(DBCHandle),b"msg_speed_can",byref(CanMsg[2]))
    #发送CAN数据
    SendedNum = CAN_SendMsg(DevHandles[0],CAN1,CanMsg,3)
    if(SendedNum >= 0):
        print("Success send frames:%d"%SendedNum)
    else:
        print("Send CAN data failed! %d"%SendedNum)
    # Delay
    sleep(0.5)
    # 读取CAN数据
    CanMsgBuffer = (CAN_MSG*10240)()
    CanNum = CAN_GetMsg(DevHandles[0],CAN2,byref(CanMsgBuffer))
    if CanNum > 0:
        print("Get CanNum = %d"%CanNum)
        for i in range(0,CanNum):
            print("CanMsg[%d].ID = 0x%08X"%(i,CanMsgBuffer[i].ID))
            print("CanMsg[%d].TimeStamp = %d"%(i,CanMsgBuffer[i].TimeStamp))
            print("CanMsg[%d].Data = "%i,end='')
            for j in range(0,CanMsgBuffer[i].DataLen):
                print("%02X "%CanMsgBuffer[i].Data[j],end='')
            print("")
    elif CanNum == 0:
        print("No CAN data!")
    else:
        print("Get CAN data error!")
    #将CAN消息数据填充到信号里面
    DBC_SyncCANMsgToValue(c_uint64(DBCHandle),CanMsgBuffer,CanNum)
    #获取信号值并打印出来
    ValueStr = (c_char * 64)()
    DBC_GetSignalValueStr(c_uint64(DBCHandle),b"msg_moto_speed",b"moto_speed",ValueStr)
    print("moto_speed = %s"%bytes(ValueStr.value).decode('ascii'))
    DBC_GetSignalValueStr(c_uint64(DBCHandle),b"msg_oil_pressure",b"oil_pressure",ValueStr)
    print("oil_pressure = %s"%bytes(ValueStr.value).decode('ascii'))
    DBC_GetSignalValueStr(c_uint64(DBCHandle),b"msg_speed_can",b"speed_can",ValueStr)
    print("speed_can = %s"%bytes(ValueStr.value).decode('ascii'))
    # Close device
    ret = USB_CloseDevice(DevHandles[0])
    if(bool(ret)):
        print("Close device success!")
    else:
        print("Close device faild!")
        exit()


