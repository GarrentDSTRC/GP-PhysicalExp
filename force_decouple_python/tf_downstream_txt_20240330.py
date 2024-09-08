import math

import pandas as pd
import numpy as np
import os
from scipy import signal
import copy
import datetime
import matplotlib.pyplot as plt
decouple_mix    = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv',header=None)
decouple_matrix = np.array(decouple_mix,dtype=np.float64)                       #6x6解耦矩阵
gain_mix        = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
GAIN            = np.array(gain_mix.iloc[0:6],dtype=np.float64)               #6x1增益矩阵
EXC             = gain_mix.iloc[6].value

Path = r'force_decouple_python/raw_data/'
savepath = r'force_decouple_python/decoupled/'
pathlist = os.listdir(r'force_decouple_python/raw_data')

b, a = signal.butter(1, 0.01, 'lowpass')  # 越小越光
def process_data_and_calculate_metrics(raw_data,T):
    # Load matrices and constants
    decouple_mix = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv', header=None)
    decouple_matrix = np.array(decouple_mix, dtype=np.float64)

    gain_mix = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
    GAIN = np.array(gain_mix.iloc[0:6], dtype=np.float64)
    EXC = gain_mix.iloc[6].value

    # Constants for calculations
    thrust0 =-1.1409941423250798 #-0.748679094098#
    lift0   =0.31073077324270965 #0.27854819462#
    mz0     =-0.029385827592709895 #-0.0035267525 #
    c = 0.06
    s = 0.22
    U = 0.2

    # Filtering parameters
    b, a = signal.butter(1, 0.01, 'lowpass')

    # Process raw data
    h0_theta0 = np.array([raw_data[7], raw_data[0]])
    h0_theta0_filtered = signal.filtfilt(b, a, h0_theta0)

    # Normalize measurements
    hmin, hmax = np.amin(h0_theta0_filtered[0]), np.amax(h0_theta0_filtered[0])
    tmin, tmax = np.amin(h0_theta0_filtered[1]), np.amax(h0_theta0_filtered[1])
    h0_theta0_filtered[0] = (h0_theta0_filtered[0] - (hmax + hmin) / 2) / 1.250 * 0.1  ## M
    h0_theta0_filtered[1] = (h0_theta0_filtered[1] -(tmax + tmin) / 2) / 5.0 * 2 * np.pi ## rad

    #h0_theta0[0]          = (h0_theta0[0]         -  (hmax + hmin) / 2) / 1.25  * 0.1     # 平均归零, 长度 m,   1.247V=0.1m
    #h0_theta0[1] = (h0_theta0[1] - (tmax + tmin) / 2) /5.0*2*np.pi   # 平均归零, 弧度 rad, 5V=360deg

    # Decouple forces
    decoupled_force = decouple_matrix @ ((raw_data[1:7] * 1000.) / (EXC * GAIN))
    filtered_force = signal.filtfilt(b, a, decoupled_force)

    # Calculate thrust, lift, and moments in body coordinates
    # 将受力正方向和运动正方向设置为相同
    # 转动逆时针为正
    thrust = -filtered_force[0] * np.sin(h0_theta0_filtered[1]) + filtered_force[1] * np.cos( h0_theta0_filtered[1]) - thrust0  ## 推力方向为正
    lift   = -filtered_force[0] * np.cos(h0_theta0_filtered[1]) - filtered_force[1] * np.sin(h0_theta0_filtered[1])  - lift0    ## vy为正
    n=1000
    #print("4个",filtered_force[0][::n], np.cos(h0_theta0_filtered[1][::n]), filtered_force[1][::n], np.sin(h0_theta0_filtered[1][::n]),lift[::n])
    mz     = -filtered_force[5] - mz0                                                                                           ## 俯视图 逆时针为正
    #减小 不变
    print("零点",np.mean(thrust),np.mean(lift),np.mean(mz))

    Ct     = thrust/(0.5*1000*U*U*c*s)
    Cl     = lift/(0.5*1000*U*U*c*s)
    Cpt    = thrust * U

    dim    = h0_theta0_filtered.shape[1]
    vy2     = np.zeros(dim)
    wz2     = np.zeros(dim)
    vy2[1:dim-1] = (h0_theta0_filtered[0,2:] - h0_theta0_filtered[0,0:dim-2]) /0.004 # 2个时间步差分
    wz2[1:dim-1] = (h0_theta0_filtered[1,2:] - h0_theta0_filtered[1,0:dim-2]) /0.004

    #Pout                 =  -lift * vy2 + mz * wz2  ##
    #Cpin                 = (-lift * vy2 - mz * wz2)/ (0.5 * 1000 * c * s * U * U * U) ##
    Cp_heave             = -lift * vy2 / (0.5 * 1000 * c * s * U * U * U)  # heave输入功率
    Cp_pitch             = -mz * wz2 / (0.5 * 1000 * c * s * U * U * U)  # pitch输入功率
    Cpin                 = Cp_heave + Cp_pitch

    Ct_mean   = np.zeros(Ct.shape[0])+1.0e-9
    Cpin_mean = np.zeros(Ct.shape[0])+1.0e-9
    Eta       = np.zeros(Ct.shape[0])+1.0e-9
    Energy_eta       = np.zeros(Ct.shape[0])+1.0e-9
    Energy_eta2 = np.zeros(Ct.shape[0]) + 1.0e-9

    num_period = int(500*T)
    alpha=0
    for i in range(Ct.shape[0]-num_period-1):
        Ct_mean[i]   = np.mean( Ct[i:i+num_period] )
        Cpin_mean[i] = np.mean( Cpin[i:i+num_period] )
        Eta[i]       = Ct_mean[i]/Cpin_mean[i]

        #Energy_eta2[i] = np.mean(Pout[i:i + num_period]) / (0.5 * 1000 * U * U * U * 1.2 * 2 * c * s)

        #Energy_eta[i]=Cpin_mean[i] *c /(2*( np.amax(h0_theta0_filtered[0])- np.amin(h0_theta0_filtered[0]) ))

        if (vy2[i] < 0):
            alp=-(( h0_theta0_filtered[1][i]-math.pi) -np.arctan(vy2[i]/U))
        else :
            alp=((h0_theta0_filtered[1][i]-math.pi)-np.arctan(vy2[i]/U ))
        alpha += alp

    #np.mean(alpha), np.mean(Energy_eta[i])
    #return np.mean(Ct_mean), np.mean(Eta)
    filterd_data_aug = np.vstack((h0_theta0_filtered, filtered_force, thrust, lift, mz, vy2, wz2, Ct_mean, Cpin_mean, Eta))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    cun = np.mean(Ct_mean[:-num_period])
    # 将Eta的平均值转换为字符串，并保留两位小数
    cun = f"{cun:.2f}"
    # 更新文件名，将Eta的平均值插入到文件名中
    filename = f"force_decouple_python\\raw_data\\CT{timestamp}_{cun}.csv"

    op = pd.DataFrame(filterd_data_aug.T[:-num_period],columns=['h0','theta0','fx','fy','fz','mx','my','mz','f_thrust','f_lift','mz','vy','wz','Cd','Cpout','Eta'])
    #op.to_csv(opfile[:-4]+'b.csv')
    op.to_csv(filename)

    return np.mean(Ct_mean[:-num_period]), np.mean(Eta[:-num_period])

# Example use:
# Assume `raw_data` is an array with shape (8, N) where N is the number of data points
# result_data = your_data_acquisition_function()  # This should get your 8-channel data
# Ct_mean, Cl_mean = process_data_and_calculate_metrics(result_data)
# print("Average Thrust Coefficient (CT):", Ct_mean)
# print("Average Lift Coefficient (CL):", Cl_mean)

def process_data_and_calculate_metricsETA2(raw_data,T):
    # Load matrices and constants
    decouple_mix = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv', header=None)
    decouple_matrix = np.array(decouple_mix, dtype=np.float64)

    gain_mix = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
    GAIN = np.array(gain_mix.iloc[0:6], dtype=np.float64)
    EXC = gain_mix.iloc[6].value

    # Constants for calculations
    thrust0 =-0.748679094098#-0.717
    lift0   =0.28054819462#  0.275-0.099
    mz0     =-0.0035267525 # -0.00118-0.003
    c = 0.06
    s = 0.22
    U = 0.20

    # Filtering parameters
    b, a = signal.butter(1, 0.01, 'lowpass')

    # Process raw data
    h0_theta0 = np.array([raw_data[7], raw_data[0]])
    h0_theta0_filtered = signal.filtfilt(b, a, h0_theta0)

    # Normalize measurements
    hmin, hmax = np.amin(h0_theta0_filtered[0]), np.amax(h0_theta0_filtered[0])
    tmin, tmax = np.amin(h0_theta0_filtered[1]), np.amax(h0_theta0_filtered[1])
    h0_theta0_filtered[0] = (h0_theta0_filtered[0] - (hmax + hmin) / 2) / 1.250 * 0.1  ## M
    h0_theta0_filtered[1] = (h0_theta0_filtered[1] - (tmax + tmin) / 2) / 5.0 * 2 * np.pi ## rad

    #h0_theta0[0]          = (h0_theta0[0]         -  (hmax + hmin) / 2) / 1.25  * 0.1     # 平均归零, 长度 m,   1.247V=0.1m
    #h0_theta0[1] = (h0_theta0[1] - (tmax + tmin) / 2) /5.0*2*np.pi   # 平均归零, 弧度 rad, 5V=360deg

    # Decouple forces
    decoupled_force = decouple_matrix @ ((raw_data[1:7] * 1000.) / (EXC * GAIN))
    filtered_force = signal.filtfilt(b, a, decoupled_force)

    # Calculate thrust, lift, and moments in body coordinates
    # 将受力正方向和运动正方向设置为相同
    # 转动逆时针为正
    thrust = -filtered_force[0] * np.sin(h0_theta0_filtered[1]) + filtered_force[1] * np.cos( h0_theta0_filtered[1]) - thrust0  ## 推力方向为正
    lift   = -filtered_force[0] * np.cos(h0_theta0_filtered[1]) - filtered_force[1] * np.sin(h0_theta0_filtered[1])  - lift0    ## vy为正
    n=1000
    #print("4个",filtered_force[0][::n], np.cos(h0_theta0_filtered[1][::n]), filtered_force[1][::n], np.sin(h0_theta0_filtered[1][::n]),lift[::n])
    mz     = -filtered_force[5] - mz0                                                                                           ## 俯视图 逆时针为正
    #减小 不变
    print("零点",np.mean(thrust),np.mean(lift),np.mean(mz))

    Ct     = thrust/(0.5*1000*U*U*c*s)  # == Cd
    Cl     = lift/(0.5*1000*U*U*c*s)
    Cpt    = thrust * U

    dim    = h0_theta0_filtered.shape[1]
    vy2     = np.zeros(dim)
    wz2     = np.zeros(dim)
    vy2[1:dim-1] = (h0_theta0_filtered[0,2:] - h0_theta0_filtered[0,0:dim-2]) /0.004 # 2个时间步差分
    wz2[1:dim-1] = (h0_theta0_filtered[1,2:] - h0_theta0_filtered[1,0:dim-2]) /0.004

    Pout                 =  (lift * vy2 + mz * wz2)*1  ##
    #Cpin                 = (-lift * vy2 - mz * wz2)/ (0.5 * 1000 * c * s * U * U * U) ##
    Cp_heave             = lift * vy2 / (0.5 * 1000 * c * s * U * U * U)  # heave output功率
    Cp_pitch             = mz * wz2 / (0.5 * 1000 * c * s * U * U * U)    # pitch output功率
    Cpout                 = Cp_heave + Cp_pitch


    Eta          = np.zeros(Pout.shape[0])+1.0e-9
    #num_period = int(float(path[-14:-11])*500)
    num_period = int(500*T)
    for i in range(Pout.shape[0]-num_period-1):
        Eta[i]   = np.mean( Pout[i:i+num_period] ) / (0.5*1000*U*U*U*(np.amax(h0_theta0_filtered[0])- np.amin(h0_theta0_filtered[0]))*s)

    filterd_data_aug = np.vstack((h0_theta0_filtered, thrust, lift, mz, vy2, wz2, Ct, Cpout, Eta))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    Eta_mean = np.mean(Eta)
    # 将Eta的平均值转换为字符串，并保留两位小数
    Eta_mean_str = f"{Eta_mean:.2f}"
    # 更新文件名，将Eta的平均值插入到文件名中
    filename = f"force_decouple_python\\raw_data\\Eta{timestamp}_{Eta_mean_str}.csv"
    op = pd.DataFrame(filterd_data_aug.T[:-num_period-1],columns=['h0','theta0','f_thrust','f_lift','mz','vy','wz','Cd','Cpout','Eta'])
    op.to_csv(filename)

    return np.mean(Eta) , 0


# def process_data_and_calculate_metricsETA(raw,T):
#     #print('raw data.shape = ',raw)
#     decouple_mix = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv', header=None)
#     decouple_matrix = np.array(decouple_mix,dtype=np.float64)                       #6x6解耦矩阵
#     gain_mix = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
#     GAIN            = np.array(gain_mix.iloc[0:6],dtype=np.float64)               #6x1增益矩阵
#     EXC             = gain_mix.iloc[6].value
#
#     #Path = r'raw data/'
#     #savepath = r'decoupled/'
#     #pathlist = os.listdir(r'raw data')
#
#     b, a = signal.butter(1, 0.1, 'lowpass')
#     ## 零点方向：thrust推力方向为正(与水流反向); lift h(t)>0为正; mz 朝下为正(逆时针)
#     # 20220103  t0=-0.1676   l0=-0.154   m0=0.009
#     thrust0 = -0.06
#     lift0   = 0.093
#     mz0     = 0.0016
#
#     c      = 0.06
#     s      = 0.22
#     #h0     = 0.075*3/4
#     U      = 0.200
#     #T      = 1.25
#     #St     = float(path[:4])
#     #T      = 2*h0/(St*U)  # 2*h0/(St*U)
#     #V0     = 2*np.pi*h0/T
#     #Nt     = int(T*500)         # 一周期有多少个采集点
#     #data  = pd.read_csv(opfile,header=None)          # 读取数据 8 列
#     #raw   = np.array(data, dtype=np.float64).T       #(8, N)
#     #raw   = np.loadtxt(opfile).T                     #(8, N)
#     #tmp0=raw
#     h0_theta0  = np.array([raw[7],raw[0]])            #(2, N) h0, theta0
#     tmp0 = copy.deepcopy(h0_theta0)
#     #times = np.arange(0, raw.shape[1])
#     #plt.plot(times, tmp0[0], label='raw')
#     h0_theta0  = signal.filtfilt(b, a, h0_theta0)     #(2, N) h0, theta0. filtered
#     tmp1 = copy.deepcopy(h0_theta0)
#     #print(h0_theta0[0,1:10])
#     #plt.plot(times, tmp1[0], label = 'filtered')
#     #plt.legend()
#     #plt.show()
#
#
#
#     #hmin, hmax, tmin, tmax = np.amin(h0_theta0[0]), np.amax(h0_theta0[0]), np.amin(h0_theta0[1]), np.amax(h0_theta0[1])
#     hmin, hmax, tmin, tmax = 0.689, 2.198, 1.266, 3.385
#     h0_theta0[0] = (h0_theta0[0] - (hmax + hmin) / 2) /1.25*0.1     # 平均归零, 长度 m,   1.247V=0.1m
#     h0_theta0[1] = (h0_theta0[1] - (tmax + tmin) / 2) /5.0*2*np.pi   # 平均归零, 弧度 rad, 5V=360deg
#
#     ## 解耦
#     decoupled_force  = decouple_matrix @ ((raw[1:7]*1000.)/(EXC*GAIN))  # 解耦
#     filterd_force    = signal.filtfilt(b, a, decoupled_force)  #(6, N) fx fy fz mx my mz, 传感器坐标系
#     #filterd_data_aug = np.vstack((h0_theta0, filterd_force[1], filterd_force[0], filterd_force[5]))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz
#
#     ## 转到大地坐标系, 计算效率
#     thrust = -filterd_force[0]*np.sin(h0_theta0[1]) + filterd_force[1]*np.cos(h0_theta0[1]) - thrust0  # fx*sin(theta) - fy*cos(theta)
#     lift   = filterd_force[0]*np.cos(h0_theta0[1]) + filterd_force[1]*np.sin(h0_theta0[1]) - lift0    # fx*cos(theta) + fy*sin(theta)
#     mz     = filterd_force[5] - mz0
#     #filterd_data_aug = np.vstack((h0_theta0, thrust, lift, filterd_force[5]))  #(5, N) h0,theta0, 大地坐标系下thrust, lift, mz
#     Cd     = thrust/(0.5*1000*U*U*c*s)
#     Cpt    = thrust * U
#
#     dim    = h0_theta0.shape[1]
#     #vy     = np.zeros(dim)
#     #wz     = np.zeros(dim)
#     #vy[0:dim-int(Nt/4)] = h0_theta0[0,int(Nt/4):]
#     #wz[0:dim-int(Nt/4)] = h0_theta0[1,int(Nt/4):]
#     vy2     = np.zeros(dim)
#     wz2     = np.zeros(dim)
#     vy2[1:dim-1] = (h0_theta0[0,2:] - h0_theta0[0,0:dim-2]) /0.004 # 2个时间步差分
#     wz2[1:dim-1] = (h0_theta0[1,2:] - h0_theta0[1,0:dim-2]) /0.004
#
#     Pout                 =  lift * vy2 - mz * wz2  ##
#     Cp_heave             =  lift * vy2 / (0.5 * 1000 * c * s * U * U * U)
#     Cp_pitch             =  - mz * wz2 / (0.5 * 1000 * c * s * U * U * U)
#     Cpout                =  Cp_heave + Cp_pitch
#     ## mean
#     #Cpout_mean   = np.zeros(Cpout.shape[0])+1.0e-9
#     Eta          = np.zeros(Pout.shape[0])+1.0e-9
#     #num_period = int(float(path[-14:-11])*500)
#     num_period = int(500*T)
#     for i in range(Pout.shape[0]-num_period-1):
#         Eta[i]   = np.mean( Pout[i:i+num_period] ) / (0.5*1000*U*U*U*(np.amax(h0_theta0[0])- np.amin(h0_theta0[0]))*s)
#
#     filterd_data_aug = np.vstack((h0_theta0, thrust, lift, mz, vy2, wz2, Cd, Cpout, Eta))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz
#
#     op = pd.DataFrame(filterd_data_aug.T[:-num_period-1],columns=['h0','theta0','f_thrust','f_lift','mz','vy','wz','Cd','Cpout','Eta'])
#     #op.to_csv(opfile[:-4]+'b.csv')
#     op.to_csv('./ETA.csv')
#
#     return np.mean(Eta) , 0