"""
文件说明:USB2XXX设备操作相关函数集合
更多帮助:www.toomoss.com
使用说明:程序正常运行,需要将sdk/libs目录复制到程序目录下
"""

from ctypes import *
import platform
import os
import shutil

# 设备固件信息定义
class DEVICE_INFO(Structure):
    _fields_ = [
        ("FirmwareName", c_char*32),   # firmware name string
        ("BuildDate", c_char*32),      # firmware build date and time string
        ("HardwareVersion", c_uint),   # hardware version
        ("FirmwareVersion",c_uint),    # firmware version
        ("SerialNumber",c_uint*3),     # USB2XXX serial number
        ("Functions",c_uint)           # USB2XXX functions
    ]

# 设备固件信息定义
class HARDWARE_INFO(Structure):
    _fields_ = [
        ("McuModel", c_char*16),    # 主芯片型号
        ("ProductModel", c_char*16),# 产品型号
        ("Version", c_uint),        # 硬件版本号
        ("CANChannelNum",c_char),   # LIN通道数
        ("PWMChannelNum",c_char),   # PWM通道数，包含LIN和DO可输出的
        ("HaveCANFD",c_char),       # 是否支持CANFD功能
        ("DIChannelNum",c_char),    # 独立DI通道数，不包含LIN通道
        ("DOChannelNum",c_char),    # 独立DO通道数，不包含LIN通道
        ("HaveIsolation",c_char),   # 是否支持电磁隔离
        ("ExPowerSupply",c_char),   # 是否支持外部电源供电
        ("IsOEM",c_char),           # 是客户定制版本吗
        ("EECapacity",c_char),      # EEPROM支持容量，单位为KByte,0表示没有EEPROM
        ("SPIFlashCapacity",c_char),# Flash容量，单位为MByte,0表示没有Flash
        ("TFCardSupport",c_char),   # 是否支持TF卡安装
        ("ProductModel", c_char*12),# 生产日期
        ("USBControl",c_char),      # 支持通过USB控制
        ("SerialControl",c_char),   # 支持串口控制
        ("EthControl",c_char),      # 支持网口控制
        ("VbatChannel",c_char)      # 可以控制的VBAT输出通道数
    ]

# 定义电压输出值
POWER_LEVEL_1V8     = 1 # 输出1.8V
POWER_LEVEL_2V5     = 2 # 输出2.5V
POWER_LEVEL_3V3     = 3 # 输出3.3V

#复制库文件到程序目录下
if(not os.path.exists("libs")):
    if(os.path.exists("../../../sdk/libs")):
        shutil.copytree("../../../sdk/libs", "./libs")
    else:
        print("libs does not exist,You need to manually copy the libs directory to the current directory")
        exit()

#根据系统自动导入对应的库文件，若没能识别到正确的系统，可以修改下面的源码
if(platform.system()=="Windows"):
    if "64bit" in platform.architecture():
        windll.LoadLibrary(os.getcwd()+"/libs/windows/x86_64/libusb-1.0.dll" )
        USB2XXXLib = windll.LoadLibrary(os.getcwd()+"/libs/windows/x86_64/USB2XXX.dll" )
    else:
        windll.LoadLibrary(os.getcwd()+"/libs/windows/x86_32/libusb-1.0.dll" )
        USB2XXXLib = windll.LoadLibrary(os.getcwd()+"/libs/windows/x86_32/USB2XXX.dll" )
elif(platform.system()=="Darwin"):
    cdll.LoadLibrary(os.getcwd()+"/libs/mac_os/libusb-1.0.0.dylib" )
    USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/mac_os/libUSB2XXX.dylib" )
elif(platform.system()=="Linux"):
    if 'armv7' in platform.machine():
        cdll.LoadLibrary(os.getcwd()+"/libs/linux/armv7/libusb-1.0.so" )
        USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/armv7/libUSB2XXX.so" )
    elif 'mips64' in platform.machine():
        cdll.LoadLibrary(os.getcwd()+"/libs/linux/mips64/libusb-1.0.so" )
        USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/mips64/libUSB2XXX.so" )
    elif 'aarch64' in platform.machine():
        cdll.LoadLibrary(os.getcwd()+"/libs/linux/aarch64/libusb-1.0.so" )
        USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/aarch64/libUSB2XXX.so" )
    elif 'arm64' in platform.machine():
        cdll.LoadLibrary(os.getcwd()+"/libs/linux/arm64/libusb-1.0.so" )
        USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/arm64/libUSB2XXX.so" )
    else:
        if "64bit" in platform.architecture():
            cdll.LoadLibrary(os.getcwd()+"/libs/linux/x86_64/libusb-1.0.so" )
            USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/x86_64/libUSB2XXX.so" )
        else:
            cdll.LoadLibrary(os.getcwd()+"/libs/linux/x86/libusb-1.0.so" )
            USB2XXXLib = cdll.LoadLibrary(os.getcwd()+"/libs/linux/x86/libUSB2XXX.so" )
else:
    print("unsupported system")
    exit()

# 扫描设备并获取设备号
def USB_ScanDevice(pDevHandle):
    return USB2XXXLib.USB_ScanDevice(pDevHandle)

# 打开设备
def USB_OpenDevice(DevHandle):
    return USB2XXXLib.USB_OpenDevice(DevHandle)

# 复位设备，复位后需要调用设备重连函数
def USB_ResetDevice(DevHandle):
    return USB2XXXLib.USB_ResetDevice(DevHandle)

# 重连设备，检测到USB断开连接后可以调用该函数恢复连接
def USB_RetryConnect(DevHandle):
    return USB2XXXLib.USB_RetryConnect(DevHandle)

# 等待设备重连成功
def USB_WaitResume(DevHandle,TimeOutMs):
    return USB2XXXLib.USB_WaitResume(DevHandle,TimeOutMs)

# 获取设备固件信息
def DEV_GetDeviceInfo(DevHandle, pDevInfo, pFunctionStr):
    return USB2XXXLib.DEV_GetDeviceInfo(DevHandle, pDevInfo, pFunctionStr)

# 获取设备硬件信息
def DEV_GetHardwareInfo(DevHandle, pHardwareInfo):
    return USB2XXXLib.DEV_GetHardwareInfo(DevHandle, pHardwareInfo)

# 关闭设备，一般是在程序退出时才调用
def USB_CloseDevice(DevHandle):
    return USB2XXXLib.USB_CloseDevice(DevHandle)

def DEV_EraseUserData(DevHandle):
    return USB2XXXLib.DEV_EraseUserData(DevHandle)

def DEV_WriteUserData(DevHandle,OffsetAddr,pWriteData,DataLen):
    return USB2XXXLib.DEV_WriteUserData(DevHandle,OffsetAddr,pWriteData,DataLen)

def DEV_ReadUserData(DevHandle,OffsetAddr,pReadData,DataLen):
    return USB2XXXLib.DEV_ReadUserData(DevHandle,OffsetAddr,pReadData,DataLen)

def DEV_SetPowerLevel(DevHandle,PowerLevel):
    return USB2XXXLib.DEV_SetPowerLevel(DevHandle,PowerLevel)

def DEV_GetTimestamp(DevHandle,BusType,pTimestamp):
    return USB2XXXLib.DEV_GetTimestamp(DevHandle,BusType,pTimestamp)

def DEV_ResetTimestamp(DevHandle):
    return USB2XXXLib.DEV_ResetTimestamp(DevHandle)

# 获取库编译日期
def DEV_GetDllBuildTime(pDateTime):
    return USB2XXXLib.DEV_GetDllBuildTime(pDateTime)

