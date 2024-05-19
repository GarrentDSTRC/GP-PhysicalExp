import pandas as pd
import numpy as np
import os
from scipy import signal
import copy
import matplotlib.pyplot as plt
decouple_mix    = pd.read_csv('m8301c_use/decoupled.csv',header=None)
decouple_matrix = np.array(decouple_mix,dtype=np.float64)                       #6x6解耦矩阵
gain_mix        = pd.read_csv('m8301c_use/gain.csv')
GAIN            = np.array(gain_mix.iloc[0:6],dtype=np.float64)               #6x1增益矩阵
EXC             = gain_mix.iloc[6].value

Path = r'raw data/'
savepath = r'decoupled/'
pathlist = os.listdir(r'raw data')

b, a = signal.butter(1, 0.1, 'lowpass')
for path in pathlist:

    opfile = Path + path
    ## 零点方向：thrust推力方向为正(与水流反向); lift h(t)>0为正; mz 朝下为正(逆时针)
    # 20220103  t0=-0.1676   l0=-0.154   m0=0.009
    thrust0 = -0.06
    lift0   = 0.093
    mz0     = 0.0094

    c      = 0.06
    s      = 0.22
    #h0     = 0.075*3/4
    U      = 0.240
    T      = 1.25
    #St     = float(path[:4])
    #T      = 2*h0/(St*U)  # 2*h0/(St*U)
    #V0     = 2*np.pi*h0/T
    #Nt     = int(T*500)         # 一周期有多少个采集点
    #data  = pd.read_csv(opfile,header=None)          # 读取数据 8 列
    #raw   = np.array(data, dtype=np.float64).T       #(8, N)
    raw   = np.loadtxt(opfile).T                     #(8, N)
    #tmp0=raw
    h0_theta0  = np.array([raw[7],raw[0]])            #(2, N) h0, theta0
    tmp0 = copy.deepcopy(h0_theta0)
    #times = np.arange(0, raw.shape[1])
    #plt.plot(times, tmp0[0], label='raw')
    h0_theta0  = signal.filtfilt(b, a, h0_theta0)     #(2, N) h0, theta0. filtered
    tmp1 = copy.deepcopy(h0_theta0)
    #print(h0_theta0[0,1:10])
    #plt.plot(times, tmp1[0], label = 'filtered')
    #plt.legend()
    #plt.show()



    #hmin, hmax, tmin, tmax = np.amin(h0_theta0[0]), np.amax(h0_theta0[0]), np.amin(h0_theta0[1]), np.amax(h0_theta0[1])
    hmin, hmax, tmin, tmax = 0.689, 2.198, 1.266, 3.385
    h0_theta0[0] = (h0_theta0[0] - (hmax + hmin) / 2) /1.25*0.1     # 平均归零, 长度 m,   1.247V=0.1m
    h0_theta0[1] = (h0_theta0[1] - (tmax + tmin) / 2) /5.0*2*np.pi   # 平均归零, 弧度 rad, 5V=360deg

    ## 解耦
    decoupled_force  = decouple_matrix @ ((raw[1:7]*1000.)/(EXC*GAIN))  # 解耦
    filterd_force    = signal.filtfilt(b, a, decoupled_force)  #(6, N) fx fy fz mx my mz, 传感器坐标系
    #filterd_data_aug = np.vstack((h0_theta0, filterd_force[1], filterd_force[0], filterd_force[5]))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz

    ## 转到大地坐标系, 计算效率
    thrust = -filterd_force[0]*np.sin(h0_theta0[1]) + filterd_force[1]*np.cos(h0_theta0[1]) - thrust0  # fx*sin(theta) - fy*cos(theta)
    lift   = -filterd_force[0]*np.cos(h0_theta0[1]) - filterd_force[1]*np.sin(h0_theta0[1]) - lift0    # fx*cos(theta) + fy*sin(theta)
    mz     = filterd_force[5] - mz0
    #filterd_data_aug = np.vstack((h0_theta0, thrust, lift, filterd_force[5]))  #(5, N) h0,theta0, 大地坐标系下thrust, lift, mz
    Cd     = thrust/(0.5*1000*U*U*c*s)
    Cpt    = thrust * U

    dim    = h0_theta0.shape[1]
    #vy     = np.zeros(dim)
    #wz     = np.zeros(dim)
    #vy[0:dim-int(Nt/4)] = h0_theta0[0,int(Nt/4):]
    #wz[0:dim-int(Nt/4)] = h0_theta0[1,int(Nt/4):]
    vy2     = np.zeros(dim)
    wz2     = np.zeros(dim)
    vy2[1:dim-1] = (h0_theta0[0,2:] - h0_theta0[0,0:dim-2]) /0.004 # 2个时间步差分
    wz2[1:dim-1] = (h0_theta0[1,2:] - h0_theta0[1,0:dim-2]) /0.004

    Pout                 =  lift * vy2 - mz * wz2  ##
    Cp_heave             =  lift * vy2 / (0.5 * 1000 * c * s * U * U * U)
    Cp_pitch             =  - mz * wz2 / (0.5 * 1000 * c * s * U * U * U)
    Cpout                =  Cp_heave + Cp_pitch
    ## mean
    #Cpout_mean   = np.zeros(Cpout.shape[0])+1.0e-9
    Eta          = np.zeros(Pout.shape[0])+1.0e-9
    #num_period = int(float(path[-14:-11])*500)
    num_period = int(500*T)
    for i in range(Pout.shape[0]-num_period-1):
        Eta[i]   = np.mean( Pout[i:i+num_period] ) / (0.5*1000*U*U*U*1.2*2*c*s)

    filterd_data_aug = np.vstack((h0_theta0, thrust, lift, mz, vy2, wz2, Cd, Cpout, Eta))  #(5, N) h0,theta0,fy(指向thrust),fx(指向lift),mz


    #print(T)
    ##  plot
    """"""
    #print(h0_theta0[0, 1:10])
    times = np.arange(0, raw.shape[1])
    plt.plot(times, tmp0[0], label='tmp0')
    plt.plot(times, tmp1[0], label='tmp1')
    plt.legend()
    plt.show()

    ##
    op = pd.DataFrame(filterd_data_aug.T,columns=['h0','theta0','f_thrust','f_lift','mz','vy','wz','Cd','Cpout','Eta'])
    #op.to_csv(opfile[:-4]+'b.csv')
    op.to_csv(savepath+path[:-4]+' D.csv')










#op.to_csv('m8301c_use/'+opfile+'b.csv')
# a = [[-0.011063,0.002213,-0.001144,0.012589,-0.011673,0.00206]]
# a = np.array(a,dtype=np.float64).T
# dat = a*1000./(EXC*GAIN)
# print(dat)
# print(dm@dat)
# print(dm@b)
# print()
# fx = plt.figure()
# fy = plt.figure()
# fz = plt.figure()
# mx = plt.figure()
# my = plt.figure()
# mz = plt.figure()
# fx.plot(np.arange(0,output[0].shape[0]),op[0])
# plt.show()
# fy.plot(np.arange(0,output[0].shape[0]),op[1])
# plt.show()
# fz.plot(np.arange(0,output[0].shape[0]),op[2])
# plt.show()
# mx.plot(np.arange(0,output[0].shape[0]),op[3])
# plt.show()
# my.plot(np.arange(0,output[0].shape[0]),op[4])
# plt.show()
# plt.plot(np.arange(0,output[0].shape[0]),op[5])
# plt.show()
