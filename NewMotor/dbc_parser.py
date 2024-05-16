"""
文件说明：USB2XXX LIN LDF文件解析相关函数集合
更多帮助：www.toomoss.com
"""
from ctypes import *
import platform
from usb_device import *

# 定义函数返回错误代码
DBC_PARSER_OK                =  0 #没有错误
DBC_PARSER_FILE_OPEN         =(-1)#打开文件出错
DBC_PARSER_FILE_FORMAT       =(-2)#文件格式错误
DBC_PARSER_DEV_DISCONNECT    =(-3)#设备未连接
DBC_PARSER_HANDLE_ERROR      =(-4)#DBC Handle错误
DBC_PARSER_GET_INFO_ERROR    =(-5)#获取解析后的数据出错
DBC_PARSER_DATA_ERROR        =(-6)#数据处理错误
DBC_PARSER_SLAVE_NACK        =(-7)#从机未响应数据

def DBC_ParserFile(DevHandle, pDBCFileName):
    USB2XXXLib.DBC_ParserFile.restype = c_uint64
    return USB2XXXLib.DBC_ParserFile(DevHandle, pDBCFileName)
def DBC_GetMsgQuantity(DBCHandle):
    return USB2XXXLib.DBC_GetMsgQuantity(DBCHandle)
def DBC_GetMsgName(DBCHandle, index, pMsgName):
    return USB2XXXLib.DBC_GetMsgName(DBCHandle, index, pMsgName)
def DBC_GetMsgSignalQuantity(DBCHandle, pMsgName):
    return USB2XXXLib.DBC_GetMsgSignalQuantity(DBCHandle, pMsgName)
def DBC_GetMsgSignalName(DBCHandle, pMsgName, index, pSignalName):
    return USB2XXXLib.DBC_GetMsgSignalName(DBCHandle, pMsgName, index, pSignalName)
def DBC_GetMsgPublisher(DBCHandle, pMsgName, pPublisher):
    return USB2XXXLib.DBC_GetMsgPublisher(DBCHandle, pMsgName, pPublisher)
#设置信号值
def DBC_SetSignalValue(DBCHandle, pMsgName, pSignalName, Value):
    return USB2XXXLib.DBC_SetSignalValue(DBCHandle, pMsgName, pSignalName, Value)
#获取信号值
def DBC_GetSignalValue(DBCHandle, pMsgName, pSignalName, pValue):
    return USB2XXXLib.DBC_GetSignalValue(DBCHandle, pMsgName, pSignalName, pValue)
def DBC_GetSignalValueStr(DBCHandle, pMsgName, pSignalName, pValueStr):
    return USB2XXXLib.DBC_GetSignalValueStr(DBCHandle, pMsgName, pSignalName, pValueStr)
#将CAN消息数据填充到信号里面
def DBC_SyncCANMsgToValue(DBCHandle, pCANMsg,MsgLen):
    return USB2XXXLib.DBC_SyncCANMsgToValue(DBCHandle, pCANMsg,MsgLen)
def DBC_SyncCANFDMsgToValue(DBCHandle, pCANFDMsg, MsgLen):
    return USB2XXXLib.DBC_SyncCANFDMsgToValue(DBCHandle, pCANFDMsg, MsgLen)
#将信号数据填充到CAN消息里面
def DBC_SyncValueToCANMsg(DBCHandle, pMsgName, pCANMsg):
    return USB2XXXLib.DBC_SyncValueToCANMsg(DBCHandle, pMsgName, pCANMsg)
def DBC_SyncValueToCANFDMsg(DBCHandle, pMsgName, pCANFDMsg):
    return USB2XXXLib.DBC_SyncValueToCANFDMsg(DBCHandle, pMsgName, pCANFDMsg)
