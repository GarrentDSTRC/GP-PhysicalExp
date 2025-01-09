#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a example module '

__author__ = 'ZhisuTech'

import time
import sys
#import daqlib.libdaq as libdaq
import sample.libdaq as libdaq
from ctypes import *
from sample.device import *
import math
import numpy as np

def dac_example( device ):
    #输出3个0.5V到1.0V阶跃信号
    print("output three step pulse from 0.5 to 1.0")
    dac_cfg=dac_wavepara()
    dac_cfg.buf=[0.5,1.0]
    dac_cfg.cycles=3 #the data in buf output three times
    dac_cfg.frequency=10000 #10KHz
    dac_cfg.trigger_mode=libdaq.DAC_TRIGGER_MODE_AUTO # auto trigger mode
    device.dac.set_wavepara(dac_cfg)
    time.sleep(1)
    device.dac.stop()

    #输出正弦波，同样的方法可以输出三角波，锯齿波等
    print("output 1kHz sin wave")
    points=100 #一个周期取100个点
    amplitude=1.0;#幅度,1.0V
    offset=1.2;   #偏置,1.2V
    buf=[]
    for x in range(0,points):
        #angle change to arc
        buf.append(amplitude*(math.sin(x*(2*math.pi/points)))+offset)

    dac_cfg.buf=buf
    dac_cfg.cycles=0 # Continuous output
    dac_cfg.frequency=10000  #point update frequency 10kHz,so sin wave frequency is 1kHz
    dac_cfg.trigger_mode=libdaq.DAC_TRIGGER_MODE_SOFT # soft trigger mode
    device.dac.set_wavepara(dac_cfg)
    device.dac.start()
    time.sleep(0.1)
    device.dac.stop()

def adc_config_channel(device):
    #channel config, all channels input range configed as -10~10V
    ch_range=libdaq.CHANNEL_RANGE_N5V_P5V    #-10V~10V量程，可选 CHANNEL_RANGE_N5V_P5V -5V~5V量程
    ch_couplemode=libdaq.ADC_CHANNEL_DC_COUPLE #固定，不可修改
    ch_refground=libdaq.ADC_CHANNEL_REFGND_RSE #固定，不可修改

    for channel in range(0,8):
        device.adc.config_channel_ex(channel,ch_range,ch_couplemode,ch_refground)

#设置ADC模块的通道校准参数
def adc_calibrate_channel(device):
    #增益和偏置改为自己需要的参数
    gain=1.0
    offset=0.0
    #channel参数为0xFF时，代表同时设置所有通道为相同参数
    #channel取值0-8时，设置指定通道的参数
    device.adc.calibrate_channel(0xFF,gain,offset)

#ADC模块单次采样范例
def adc_single_sample_example(device):
    #get adc data
    print("ADC single sample:")
    channel_list=[0,1,2,3,4,5,6,7]#采样通道列表，0-7，8个通道
    (errorcode,result)=device.adc.singleSample(channel_list)

    #打印采样结果
    print("channel_list:",end="")
    print(channel_list)
    print("result:",end="")
    for data in result :
        sys.stdout.write("%2.4f " % (data))
    print("")



#有限点采样,软件触发
def adc_continuous_sampling(device, duration):
    samplepara=adc_samplepara()
    channel_list=[0,1,2,3,4,5,6,7]#采样通道列表，0-7，8个通道
    samplepara.channel_list=channel_list
    samplepara.sample_mode=ADC_SAMPLE_MODE_SYNC   #只支持同步采样模式
    samplepara.frequency=500  #采样频率10kHz
    samplepara.cycles=5 #采样5组数据

    device.adc.select_triggerSrc(ADC_TRIG_SRC_SW)#选择软件触发方式
    device.adc.set_sample_parameter(samplepara) #配置采样参数


    data_len=samplepara.cycles*len(channel_list) #计算数据个数
    start_time = time.time()
    collected_data = []
    while (time.time() - start_time) < duration:
        device.adc.clear_buffer()  # 清除ADC模块硬件和软件缓冲区数据
        device.adc.start_task()  # 启动采样任务
        device.adc.send_trigger()  # 默认为软件触发,发送软件触发命令
        (errorcode,result)=device.adc.read_analog_sync(data_len,1000)
        collected_data.extend(result)
        device.adc.stop_task()
        #打印采样结果
    device.adc.stop_task()  # 停止采样任务
    print("ADC sample, soft trigger, get data len: %d"%(len(collected_data)))

    reshaped_data = np.reshape(collected_data, (len(collected_data) // 8, 8))
    # 转置数组，使得列数变为8
    transposed_data = reshaped_data.T
    final_list = transposed_data.tolist()
    return transposed_data

#有限点采样,硬件触发
def adc_hw_trigger_example( device):
    samplepara=adc_samplepara()
    channel_list=[0,1,2,3,4,5,6,7]#采样通道列表，0-7，8个通道
    samplepara.channel_list=channel_list
    samplepara.sample_mode=ADC_SAMPLE_MODE_SYNC   #只支持同步采样模式
    samplepara.frequency=10000  #采样频率10kHz
    samplepara.cycles=5 #采样5组数据


    #配置数字触发源，电平触发类型，高电平触发方式；
    #也可以选择模拟触发源，模拟触发的信号源为AI0通道，触发电平由AO控制
    device.adc.config_triggerSrc(ADC_TRIG_SRC_HWD,0,ADC_TRIG_TYPE_LEVEL,ADC_TRIG_EDGE_RISE,ADC_TRIG_LEVEL_HIGH,0)
    device.adc.select_triggerSrc(ADC_TRIG_SRC_HWD);#选择数字触发源
    device.adc.set_sample_parameter(samplepara) #配置采样参数
    device.adc.clear_buffer() #清除ADC模块硬件和软件缓冲区数据
    device.adc.start_task()   #启动采样任务

    data_len=samplepara.cycles*len(channel_list) #计算数据个数
    (errorcode,result)=device.adc.read_analog_sync(data_len,10000)
    device.adc.stop_task()#停止采样任务

    #打印采样结果
    print("ADC sample, soft trigger, get data len: %d"%(len(result)))
    for ch_index in range(0,len(channel_list)):
        ch_data=device.adc.extractChannelData(result,len(channel_list),ch_index)
        sys.stdout.write("ch_%02d: " % (ch_index))
        for data in ch_data :
            sys.stdout.write("%2.4f " % (data))
        print("")


def main_sample(duration,T):
    index=0
    libdaq.libdaq_init()
    device_count=libdaq.libdaq_device_get_count()
    if(device_count < 0 ):
        print("no device detected!\n")
        return
     
    (errorcode,device_name)=libdaq.libdaq_device_get_name(index)
    print("device: %s deteced"%(device_name))

    device=libdaq.DAQUSB401x(device_name)
    
    #adc example
    adc_config_channel(device)  #配置ADC模块的采样通道参数
    adc_calibrate_channel(device)#设置ADC模块的通道校准参数,可选
    samplepara=adc_samplepara()
    channel_list=[0,1,2,3,4,5,6,7]#采样通道列表，0-7，8个通道
    samplepara.channel_list=channel_list
    samplepara.sample_mode=ADC_SAMPLE_MODE_SYNC   #只支持同步采样模式
    samplepara.frequency=500  #采样频率10kHz
    samplepara.cycles=int(16*500*T) #采样5组数据
    print('samplepara.cycles',samplepara.cycles)

    device.adc.select_triggerSrc(ADC_TRIG_SRC_SW)#选择软件触发方式
    device.adc.set_sample_parameter(samplepara) #配置采样参数


    data_len=samplepara.cycles*len(channel_list) #计算数据个数
    start_time = time.time()
    collected_data = []
    while (time.time() - start_time) < duration:
        device.adc.clear_buffer()  # 清除ADC模块硬件和软件缓冲区数据
        device.adc.start_task()  # 启动采样任务
        device.adc.send_trigger()  # 默认为软件触发,发送软件触发命令
        (errorcode,result)=device.adc.read_analog_sync(data_len,1000000)
        collected_data.extend(result)
        device.adc.stop_task()
        #打印采样结果
        device.adc.stop_task()  # 停止采样任务
        print("ADC sample, soft trigger, get data len: %d"%(len(collected_data)))

    reshaped_data = np.reshape(collected_data, (len(collected_data) // 8, 8))
    # 转置数组，使得列数变为8
    transposed_data = reshaped_data.T
    final_list = transposed_data.tolist()

    print("Sample finish!")
    return transposed_data


