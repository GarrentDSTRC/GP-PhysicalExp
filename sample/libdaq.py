#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
  @file    libdaq.py
  @author  libdaq Library Team,support@zishuech.com
  @version V2.2.0
  @date    2020-10-28
  @brief   This file provide all libdaq API functions and definitions wrapped for python language.
 
  Copyright (c) 2016-2020 ZishuTech.Co.Ltd. All rights reserved.
 
'''
__author__ = 'ZhisuTech'


import ctypes
from ctypes import byref
import platform
import os
# detect python version
architecture=platform.architecture()

#  load dll
dir=os.path.dirname(os.path.abspath(__file__))
if architecture[0] == '64bit':
    dll_path = os.path.join(dir,'./daqlib/MS64/daqlib.dll')
else:
    dll_path = os.path.join(dir,'./daqlib/MS32/daqlib.dll')

daqdll = ctypes.cdll.LoadLibrary(dll_path)

# API for library
_libdaq_init = daqdll.libdaq_init
_libdaq_exit = daqdll.libdaq_exit
_libdaq_set_option = daqdll.libdaq_set_option
_libdaq_get_version =daqdll.libdaq_get_version
_libdaq_get_error_desc = daqdll.libdaq_get_error_desc
_libdaq_get_error_str  = daqdll.libdaq_get_error_str

# API for devices
_libdaq_device_get_count = daqdll.libdaq_device_get_count
_libdaq_device_rename_byindex = daqdll.libdaq_device_rename_byindex
_libdaq_device_rename_byname = daqdll.libdaq_device_rename_byname
_libdaq_device_get_name = daqdll.libdaq_device_get_name
_libdaq_device_get_version=daqdll.libdaq_device_get_version
_libdaq_device_setUID_byindex=daqdll.libdaq_device_setUID_byindex
_libdaq_device_setUID_byname=daqdll.libdaq_device_setUID_byname

# API for GPIO
_libdaq_gpio_get_iocount=daqdll.libdaq_gpio_get_iocount
_libdaq_gpio_get_ioattrs=daqdll.libdaq_gpio_get_ioattrs
_libdaq_gpio_get_config=daqdll.libdaq_gpio_get_config
_libdaq_gpio_set_config=daqdll.libdaq_gpio_set_config
_libdaq_gpio_write_bit = daqdll.libdaq_gpio_write_bit
_libdaq_gpio_write_port = daqdll.libdaq_gpio_write_port
_libdaq_gpio_read_bit = daqdll.libdaq_gpio_read_bit
_libdaq_gpio_read_port = daqdll.libdaq_gpio_read_port

# API for DAC
_libdaq_dac_set_wavepara = daqdll.libdaq_dac_set_wavepara
_libdaq_dac_set_value=daqdll.libdaq_dac_set_value
_libdaq_dac_start = daqdll.libdaq_dac_start
_libdaq_dac_stop = daqdll.libdaq_dac_stop

# API for ADC
_libdaq_adc_config_channel=daqdll.libdaq_adc_config_channel
_libdaq_adc_config_channel_ex=daqdll.libdaq_adc_config_channel_ex
_libdaq_adc_calibrate_channel=daqdll.libdaq_adc_calibrate_channel
_libdaq_adc_singleSample=daqdll.libdaq_adc_singleSample
_libdaq_adc_set_sample_parameter    = daqdll.libdaq_adc_set_sample_parameter
_libdaq_adc_set_sample_parameter_ex = daqdll.libdaq_adc_set_sample_parameter_ex
_libdaq_adc_clear_buffer=daqdll.libdaq_adc_clear_buffer
_libdaq_adc_read_analog = daqdll.libdaq_adc_read_analog
_libdaq_adc_read_analog_sync = daqdll.libdaq_adc_read_analog_sync
_libdaq_adc_send_trigger=daqdll.libdaq_adc_send_trigger
_libdaq_adc_stop=daqdll.libdaq_adc_stop
_libdaq_adc_start_task=daqdll.libdaq_adc_start_task
_libdaq_adc_stop_task=daqdll.libdaq_adc_stop_task
_libdaq_adc_config_triggerSrc=daqdll.libdaq_adc_config_triggerSrc
_libdaq_adc_select_triggerSrc=daqdll.libdaq_adc_select_triggerSrc
_libdaq_adc_extractChannelData=daqdll.libdaq_adc_extractChannelData
_libdaq_adc_set_realtime=daqdll.libdaq_adc_set_realtime


class hw_version_c(ctypes.Structure):
    _fields_ = [("firmware_major",ctypes.c_uint8),  #firmware major version
                ("firmware_minor", ctypes.c_uint8), #firmware minor version
                ("firmware_micro", ctypes.c_uint8), #firmware micro version
                ("pcb_ver", ctypes.c_uint8),        #PCB version
                ("bom_ver",ctypes.c_uint8)]         #BOM version

class ioattr_c(ctypes.Structure):
    _fields_ = [("iomode", ctypes.c_uint8)]

class dac_wavepara_c(ctypes.Structure):
    _fields_ = [("buf", ctypes.POINTER(ctypes.c_double)),
                ("buflen", ctypes.c_uint16),
                ("cycles", ctypes.c_uint32),
                ("frequency", ctypes.c_double),
                ("trigger_mode", ctypes.c_uint8)]

class adc_channelpara_c(ctypes.Structure):
    _fields_=[("channel",ctypes.c_uint8),
              ("range",ctypes.c_uint8),
              ("couplemode",ctypes.c_uint8),
              ("refground",ctypes.c_uint8) ]

class adc_samplepara_c(ctypes.Structure):
    _fields_=[("channel_list",ctypes.POINTER(ctypes.c_uint8)),
              ("channel_count",ctypes.c_uint8),
              ("sample_mode",ctypes.c_uint8), #sequence or group mode,sync mode
              ("frequency",ctypes.c_uint),  #sample rate (Hz)
              ("cycles",ctypes.c_uint),     #0:continues
              ("group_interval",ctypes.c_uint)] #only used in group mode(us)


'''
    libdaq wrapped definition
'''

# libdaq access mode
LIBDAQ_ACCESS_MODE_DEV_NAME_SN_BOTH =0x00  #defaut，both device name and SN can bt used 
LIBDAQ_ACCESS_MODE_DEV_NAME_ONLY    =0x01  #only device name can bt used  
LIBDAQ_ACCESS_MODE_DEV_SN_ONLY      =0x02  #only device SN can bt used 

# UID state value
UID_OFF = 0
UID_ON  = 1

# iomode 
IOMODE_IN=0        # input only
IOMODE_OUT=1       # output only
IOMODE_IN_OUT=2    # can be configured input or output
IOMODE_IN_AF=3     # can be configured input or Alternate function
IOMODE_OUT_AF=4    # can be configured output or Alternate function
IOMODE_IN_OUT_AF=5 # can be configured input, output or Alternate function

# ioconf
IOCONF_IN=0
IOCONF_OUT=1
IOCONF_AF=2

# channel range
CHANNEL_RANGE_0_P1V             =4  #range:0-1V
CHANNEL_RANGE_0_P2V             =8  #range:0-2V
CHANNEL_RANGE_0_P2V5            =12 #range:0-2.5V
CHANNEL_RANGE_0_P5V             =16 #range:0-5V
CHANNEL_RANGE_0_P10V            =20 #range:0-10V
CHANNEL_RANGE_N78mV125_P78mV125 =81 #range:+/-78.125mV
CHANNEL_RANGE_N156mV25_P156mV25 =82 #range:+/-156.25mV
CHANNEL_RANGE_N312mV5_P312mV5   =83 #range:+/-312.5mV
CHANNEL_RANGE_N0V256_P0V256     =22 #range:+/-0.256V
CHANNEL_RANGE_N0V512_P0V512     =23 #range:+/-0.512V
CHANNEL_RANGE_N0V625_P0V625     =84 #range:+/-0.625V
CHANNEL_RANGE_N1V_P1V           =24 #range:+/-1V
CHANNEL_RANGE_N1V024_P1V024     =25 #range:+/-1.024V
CHANNEL_RANGE_N1V25_P1V25   	=26 #range:+/-1.25V
CHANNEL_RANGE_N2V_P2V           =28 #range:+/-2V
CHANNEL_RANGE_N2V048_P2V048     =29 #range:+/-2.048V
CHANNEL_RANGE_N2V5_P2V5         =32 #range:+/-2.5V
CHANNEL_RANGE_N4V096_P4V096     =33 #range:+/-4.096V
CHANNEL_RANGE_N5V_P5V           =36 #range:+/-5V
CHANNEL_RANGE_N10V_P10V         =40 #range:+/-10V
CHANNEL_RANGE_0_P20mA           =64 #range:0-20mA
CHANNEL_RANGE_N20mA_P20mA       =65 #range:+/-20mA
CHANNEL_RANGE_0_P40mA           =66 #range:0-40mA
CHANNEL_RANGE_N40mA_P40mA       =67 #range:+/-40mA

# dac trigger mode
DAC_TRIGGER_MODE_AUTO=0x00  # auto start
DAC_TRIGGER_MODE_SOFT=0x01  # soft trigger
DAC_TRIGGER_MODE_HARD=0x02  # hard trigger,now is not supported

# adc sample mode
ADC_SAMPLE_MODE_SEQUENCE  = 0x00 #sequence mode
ADC_SAMPLE_MODE_GROUP     = 0x01 #group mode
ADC_SAMPLE_MODE_SYNC      = 0x02 #sync mode

# adc channel couple mode
ADC_CHANNEL_DC_COUPLE =0x00 #DC couple
ADC_CHANNEL_AC_COUPLE =0x01 #AC couple

ADC_CHANNEL_REFGND_RSE  =0x00 #referenced single ended
ADC_CHANNEL_REFGND_NRSE =0x01 #non referenced single ended
ADC_CHANNEL_REFGND_DIFF =0x02 #differencial

#ADC_TRIG_SRC_SW config
#trigger_source
ADC_TRIG_SRC_SW  =0x00
ADC_TRIG_SRC_HWD =0x01
ADC_TRIG_SRC_HWA =0x02
ADC_TRIG_SRC_ANY =0x03
#trigger_type
ADC_TRIG_TYPE_EDGE =0x00
ADC_TRIG_TYPE_LEVEL =0x01
#trigger_edge
ADC_TRIG_EDGE_RISE  =0x00
ADC_TRIG_EDGE_FALL  =0x01
ADC_TRIG_EDGE_BOTH  =0x02
#trigger_level
ADC_TRIG_LEVEL_HIGH =0x00
ADC_TRIG_LEVEL_LOW  =0x01

class libdaq_version:
    major_ver=0
    minor_ver=0
    micro_ver=0

class hw_version:
    firmware_major=0   #firmware major version
    firmware_minor=0   #firmware minor version
    firmware_micro=0   #firmware micro version
    pcb_ver=0          #PCB version
    bom_ver=0          #BOM version

class ioattr:
    iomode=0

class dac_wavepara:
    buf=[]
    cycles=0
    frequency=0
    trigger_mode=0

class adc_channelpara:
    channel=0
    range=0
    couplemode=0
    refground=0

class adc_samplepara:
    channel_list=[]
    sample_mode=0
    frequency=0
    cycles=0
    group_interval=0

def libdaq_init():
    """
	libdaq init function, must be call before call any API 
	Args: None
	Returns: errorcode
	Raises: None
	"""
    errorcode = _libdaq_init()
    return errorcode

def libdaq_exit():
    """
    libdaq library exit, called before exit application to release resource
	Args: None
	Returns: None
	"""
    _libdaq_exit()

def libdaq_set_option(option):
    """
	set libdaq access mode
	Args: option, 
        LIBDAQ_ACCESS_MODE_DEV_NAME_SN_BOTH : both device name and SN can bt used 
        LIBDAQ_ACCESS_MODE_DEV_NAME_ONLY : only device name can bt used  
        LIBDAQ_ACCESS_MODE_DEV_SN_ONLY   : only device SN can bt used 
	Returns: None
	"""
    _libdaq_set_option(option)
'''
def libdaq_get_error_desc(error_code):
    rst= _libdaq_get_error_desc(error_code)
    return ctypes.string_at(rst).decode("utf-8")

def libdaq_get_error_str(error_code):
    rst= libdaq_get_error_str(error_code)
    return ctypes.string_at(rst).decode("utf-8")
'''

def libdaq_get_version():
    """
	get libdaq library version
	Args: None
	Returns: [error_code,libdaq_version]
	"""
    major_ver = ctypes.c_byte(0)
    minor_ver = ctypes.c_byte(0)
    micro_ver = ctypes.c_byte(0)
    error_code =_libdaq_get_version(ctypes.byref(major_ver),ctypes.byref(minor_ver),ctypes.byref(micro_ver))
    _version = libdaq_version()
    _version.major_ver=major_ver.value
    _version.minor_ver=minor_ver.value
    _version.micro_ver=micro_ver.value

    return [error_code,_version]

# functions for device
def libdaq_device_get_count():
    """
	get DAQ device count in PC 
	Args: None
	Returns: int Type, daq device count
	"""
    return _libdaq_device_get_count()

def libdaq_device_get_name(index):
    """
	get DAQ device name, 
	Args: 	device index 
	Returns: errorcode,device_name
	Raises: None
	"""
    device_name = ctypes.create_string_buffer(b'0', 255) # default max device name length is 255

    errorcode = _libdaq_device_get_name(index, ctypes.byref(device_name), 100)
    return errorcode,device_name.value

def libdaq_device_rename_byindex(index, newname):
    """
	rename DAQ device by index
	Args: index
		  newname
	Returns: errorcode
	Raises: None
	"""
    errorcode = _libdaq_device_rename_byindex(index, newname)
    return errorcode

def libdaq_device_rename_byname(oldname, newname):
    """
	rename DAQ device by old name of DAQ decice
	Args: 	oldname
			newname
	Returns: errorcode
	Raises: None
	"""
    '''
        if not isinstance(oldname, (str, bytes)):
            raise TypeError("oldname type must be str or bytes")
        if not isinstance(newname, (str, bytes)):
            raise TypeError("newname type must be str or bytes")
    '''
    errorcode = _libdaq_device_rename_byname(oldname, newname)
    return errorcode

def libdaq_device_get_version(device_name):
    """
	get DAQ decice version
	Args: 	device_name
			version
	Returns: errorcode,hw_version
	Raises: None
	"""
    _hw_version_c=hw_version_c() 
    errorcode = _libdaq_device_get_version(device_name, ctypes.byref(_hw_version_c))
    _hw_version=hw_version()
    _hw_version.firmware_major=_hw_version_c.firmware_major
    _hw_version.firmware_minor=_hw_version_c.firmware_minor
    _hw_version.firmware_micro=_hw_version_c.firmware_micro
    _hw_version.pcb_ver=_hw_version_c.pcb_ver
    _hw_version.bom_ver=_hw_version_c.bom_ver

    return errorcode,_hw_version

def libdaq_device_setUID_byindex(index,state):
    """
	set UID LED state,when state=UID_ON,UID LED will turn on`
	Args: 	index
			state
	Returns: errorcode
	Raises: None
	"""
    if not isinstance(index, int):
        raise TypeError("index  type must be int")
    if not state in [UID_ON,UID_OFF]:
        raise TypeError("state  type must be value of UID_state")
    
    errorcode=_libdaq_device_setUID_byindex(index,ctypes.c_uint8(state))
    return errorcode

def libdaq_device_setUID_byname(device_name,state):
    """
	set UID LED state,when state=UID_ON,UID LED will turn on`
	Args: device_name
			state
	Returns: errorcode
	Raises: None
	"""

    if not state in [UID_ON,UID_OFF]:
        raise TypeError("state  type must be value of UID_state")		
    errorcode=_libdaq_device_setUID_byname(device_name,ctypes.c_uint8(state))
    return errorcode


class libdaq_gpio(object):
    def __init__(self, device_name, module_name):
        self.__device_name = device_name
        self.__module_name = module_name
        (errorcode,self.__io_count)=self.get_iocount()

    def get_iocount(self):
        iocount=ctypes.c_uint8(0)
        errorcode=_libdaq_gpio_get_iocount(self.__device_name,self.__module_name,ctypes.byref(iocount))
        return errorcode,iocount.value

    def write_bit(self,ioIndex,BitVal):
        if not isinstance(ioIndex, int):
            raise TypeError("ioIndex  type must be int")

        if not isinstance(BitVal, int):
            raise TypeError("BitVal  type must be int")
        _BitVal=ctypes.c_uint8(BitVal)
        errorcode=_libdaq_gpio_write_bit(self.__device_name, self.__module_name, ioIndex, _BitVal)
        return errorcode

    def write_port(self,PortVal):
        if not isinstance(PortVal, list):
            raise TypeError("PortVal  type must be list")

        if len(PortVal) < self.__io_count:
            raise ValueError('input argurment PortVal size must same as iocount!')

        type_c_uint8_array_IO=ctypes.c_uint8*self.__io_count
        _PortVal=type_c_uint8_array_IO(*PortVal) # 操作符 * 展开参数 
        errorcode=_libdaq_gpio_write_port(self.__device_name,self.__module_name,ctypes.byref(_PortVal))
        return errorcode
        
    def read_bit(self,ioIndex):
        BitVal=ctypes.c_uint8(0)
        errorcode=_libdaq_gpio_read_bit(self.__device_name,self.__module_name,ioIndex,ctypes.byref(BitVal))
        return errorcode,BitVal.value
        
    def read_port(self):
        type_c_uint8_array_IO=ctypes.c_uint8*self.__io_count
        _PortVal=type_c_uint8_array_IO()
        errorcode=_libdaq_gpio_read_port(self.__device_name,self.__module_name,ctypes.byref(_PortVal))
        return errorcode , list(_PortVal)

class libdaq_dac(object):
    def __init__(self, device_name, module_name):
        self.__device_name = device_name
        self.__module_name = module_name

    def set_wavepara_ex(self, wave_buf, cycles, frequency, trigger_mode):
        """
        libdaq dac config API
        Args:
            wave_buf:dac wave data list
            cycles,
            frequency,
            startmode
        Returns:
        Raises:
        """
        if not isinstance(wave_buf, list):
            raise TypeError("wave_buf  type must be list")

        dac_cfg=dac_wavepara_c()        
        type_double_arrary=ctypes.c_double*len(wave_buf) # double array
        
        buf=type_double_arrary(*wave_buf)
        buf_p=ctypes.POINTER(ctypes.c_double)()
        buf_p.contents=buf

        dac_cfg.buf=buf_p
        dac_cfg.buflen=len(wave_buf)
        dac_cfg.cycles=cycles #the data in buf output three times
        dac_cfg.frequency=frequency #10KHz
        dac_cfg.trigger_mode=trigger_mode # auto trigger mode

        errorcode = _libdaq_dac_set_wavepara(self.__device_name,self.__module_name,ctypes.byref(dac_cfg))
        return errorcode

    def set_wavepara(self, wavepara):
        """
        libdaq dac config API
        Args:
            dac_wavepara
        Returns:
        Raises:
        """
        if not isinstance(wavepara, dac_wavepara):
            raise TypeError("wavepara  type must be dac_wavepara")

        dac_cfg=dac_wavepara_c()        
        type_double_arrary=ctypes.c_double*len(wavepara.buf) # double array
        
        buf=type_double_arrary(*wavepara.buf)
        buf_p=ctypes.POINTER(ctypes.c_double)()
        buf_p.contents=buf

        dac_cfg.buf=buf_p
        dac_cfg.buflen=len(wavepara.buf)
        dac_cfg.cycles=wavepara.cycles #the data in buf output three times
        dac_cfg.frequency=wavepara.frequency #10KHz
        dac_cfg.trigger_mode=wavepara.trigger_mode # auto trigger mode

        errorcode = _libdaq_dac_set_wavepara(self.__device_name,self.__module_name,ctypes.byref(dac_cfg))
        return errorcode

    def set_value(self,value):
        _value = ctypes.c_double(value)
        errorcode =_libdaq_dac_set_value(self.__device_name,self.__module_name,_value)
        return errorcode

    def start(self):
        errorcode =_libdaq_dac_start(self.__device_name,self.__module_name)
        return errorcode

    def stop(self):
        errorcode =_libdaq_dac_stop(self.__device_name,self.__module_name)
        return errorcode


class libdaq_adc(object):
    def __init__(self, device_name, module_name):
        self.__device_name = device_name
        self.__module_name = module_name
    
    def config_channel_ex(self,channel,chrange,couplemode,refground):
        _channel = ctypes.c_ubyte(channel)
        _chrange = ctypes.c_ubyte(chrange)
        _couplemode = ctypes.c_ubyte(couplemode)
        _refground  = ctypes.c_ubyte(refground)
        errorcode=_libdaq_adc_config_channel_ex(self.__device_name,self.__module_name, _channel, _chrange, _couplemode, _refground)
        return errorcode

    def calibrate_channel(self,channel,gain, offset):
        _channel = ctypes.c_ubyte(channel)
        _gain = ctypes.c_double(gain)
        _offset = ctypes.c_double(offset)
        errorcode=_libdaq_adc_calibrate_channel(self.__device_name,self.__module_name, _channel, _gain, _offset)
        return errorcode

    def singleSample(self,channel_list):
        ch_len=len(channel_list)
        type_uint8_arrary=ctypes.c_uint8*ch_len # uint8 array,
        channel_list=type_uint8_arrary(*channel_list)
        channel_list_p=ctypes.POINTER(ctypes.c_uint8)()
        channel_list_p.contents=channel_list

        type_double_arrary=ctypes.c_double*ch_len # uint8 array,
        result_buf=type_double_arrary(0)
        result_buf_p=ctypes.POINTER(ctypes.c_double)()
        result_buf_p.contents=result_buf

        errorcode=_libdaq_adc_singleSample(self.__device_name,self.__module_name,channel_list_p, ch_len, result_buf_p)
        return errorcode,list(result_buf)

    def set_sample_parameter_ex(self,channel_list,sample_mode,frequency,cycles,group_interval):

        channel_count=len(channel_list)
        type_uint8_arrary=ctypes.c_uint8*channel_count # uint8 array,
        channel_list=type_uint8_arrary(*channel_list)
        channel_list_p=ctypes.POINTER(ctypes.c_uint8)()
        channel_list_p.contents=channel_list

        _channel_count=ctypes.c_uint(channel_count)
        _sample_mode=ctypes.c_uint(sample_mode)
        _frequency=ctypes.c_int(frequency)
        _cycles=ctypes.c_int(cycles)
        _group_interval=ctypes.c_int(group_interval)

        errorcode=_libdaq_adc_set_sample_parameter_ex(self.__device_name,self.__module_name,channel_list_p,_channel_count,_sample_mode,_frequency,_cycles,_group_interval)
        return errorcode

    def set_sample_parameter(self,adc_samplepara):

        _samplepara_c=adc_samplepara_c()

        channel_count=len(adc_samplepara.channel_list)
        type_uint8_arrary=ctypes.c_uint8*channel_count # uint8 array,
        channel_list=type_uint8_arrary(*adc_samplepara.channel_list)
        channel_list_p=ctypes.POINTER(ctypes.c_uint8)()
        channel_list_p.contents=channel_list

        _samplepara_c.channel_list=channel_list_p
        _samplepara_c.channel_count=channel_count
        _samplepara_c.sample_mode=adc_samplepara.sample_mode
        _samplepara_c.frequency=adc_samplepara.frequency
        _samplepara_c.cycles= adc_samplepara.cycles
        _samplepara_c.group_interval=adc_samplepara.group_interval

        errorcode=_libdaq_adc_set_sample_parameter(self.__device_name,self.__module_name,ctypes.byref(_samplepara_c))
        return errorcode


    def clear_buffer(self):
        errorcode=_libdaq_adc_clear_buffer(self.__device_name,self.__module_name)
        return errorcode

    def read_analog(self,datalen):
        actual_len=ctypes.c_uint(0)
        _datalen=ctypes.c_uint(datalen)

        type_double_arrary=ctypes.c_double*datalen # double array,
        data_buf=type_double_arrary(0)
        data_buf_p=ctypes.POINTER(ctypes.c_double)()
        data_buf_p.contents=data_buf

        errorcode=_libdaq_adc_read_analog(self.__device_name,self.__module_name, data_buf_p, _datalen, ctypes.byref(actual_len))
        result=list(data_buf[0:actual_len.value])
        return errorcode, result

    def read_analog_sync(self,datalen,timeout):
        actual_len=ctypes.c_uint(0)
        _datalen=ctypes.c_uint(datalen)

        type_double_arrary=ctypes.c_double*datalen # double array,
        data_buf=type_double_arrary(0)
        data_buf_p=ctypes.POINTER(ctypes.c_double)()
        data_buf_p.contents=data_buf

        errorcode=_libdaq_adc_read_analog_sync(self.__device_name,self.__module_name, data_buf_p, _datalen, ctypes.byref(actual_len),ctypes.c_int(timeout))
        result=list(data_buf[0:actual_len.value])
        return errorcode, result

    def send_trigger(self):
        errorcode=_libdaq_adc_send_trigger(self.__device_name,self.__module_name)
        return errorcode   

    def stop(self):
        errorcode=_libdaq_adc_stop(self.__device_name,self.__module_name)
        return errorcode   
    
    def start_task(self):
        errorcode=_libdaq_adc_start_task(self.__device_name,self.__module_name)
        return errorcode

    def stop_task(self):
        errorcode=_libdaq_adc_stop_task(self.__device_name,self.__module_name)
        return errorcode

    def set_realtime(self,realtime_ms):
        _realtime_ms = ctypes.c_int(realtime_ms)
        errorcode=_libdaq_adc_set_realtime(self.__device_name,self.__module_name,_realtime_ms)
        return errorcode

    def config_triggerSrc(self, trigger_source, trigger_channel,trigger_type,trigger_edge,trigger_level,trigger_delay):
        _trigger_source=ctypes.c_ubyte(trigger_source) 
        _trigger_channel=ctypes.c_ubyte(trigger_channel) 
        _trigger_type=ctypes.c_ubyte(trigger_type) 
        _trigger_edge=ctypes.c_ubyte(trigger_edge) 
        _trigger_level=ctypes.c_ubyte(trigger_level) 
        _trigger_delay=ctypes.c_ubyte(trigger_delay) 
        errorcode=_libdaq_adc_config_triggerSrc(self.__device_name,self.__module_name,_trigger_source, _trigger_channel,_trigger_type,_trigger_edge,_trigger_level,_trigger_delay)
        return errorcode

    def select_triggerSrc(self, trigger_source):
        _trigger_source=ctypes.c_ubyte(trigger_source) 
        errorcode=_libdaq_adc_select_triggerSrc(self.__device_name,self.__module_name,_trigger_source)
        return errorcode

    def extractChannelData(self,all_data, ch_listlen,ch_index):
        all_datalen=len(all_data)
        ch_data=all_data[ch_index:all_datalen:ch_listlen]
        return ch_data

class DAQUSB3212(object):
    def __init__(self, device_name):
        self.__device_name = device_name
        self.gpioin=libdaq_gpio(self.__device_name,b'GPIOIN')
        self.gpioout=libdaq_gpio(self.__device_name,b'GPIOOUT')
        self.dac1=libdaq_dac(self.__device_name,b'DAC1')
        self.dac2=libdaq_dac(self.__device_name,b'DAC2')
        self.adc1=libdaq_adc(self.__device_name,b'ADC1')
        pass

class DAQUSB3213(DAQUSB3212):
    pass

class DAQUSB3214(DAQUSB3212):
    pass

class DAQUSB1140(object):
    def __init__(self, device_name):
        self.__device_name = device_name
        self.gpioin=libdaq_gpio(self.__device_name,b'GPIOIN')
        self.gpioout=libdaq_gpio(self.__device_name,b'GPIOOUT')
        self.dac1=libdaq_dac(self.__device_name,b'DAC1')
        self.dac2=libdaq_dac(self.__device_name,b'DAC2')
        self.adc1=libdaq_adc(self.__device_name,b'ADC1')
        self.adc2=libdaq_adc(self.__device_name,b'ADC2')
        pass

class DAQUSB1141(DAQUSB1140):
    pass


class DAQUSB401x(object):
    def __init__(self, device_name):
        self.__device_name = device_name
        self.dac=libdaq_dac(self.__device_name,b'DAC')
        self.adc=libdaq_adc(self.__device_name,b'ADC')
        pass

if __name__ == '__main__':
    libdaq_init()
    errorcode,device_name=libdaq_device_get_name(0)
    print(device_name)
    [error_code,version]=libdaq_get_version()
    print(version)