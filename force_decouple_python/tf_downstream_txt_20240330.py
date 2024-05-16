import math

import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
decouple_mix    = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv',header=None)
decouple_matrix = np.array(decouple_mix,dtype=np.float64)                       #6x6解耦矩阵
gain_mix        = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
GAIN            = np.array(gain_mix.iloc[0:6],dtype=np.float64)               #6x1增益矩阵
EXC             = gain_mix.iloc[6].value

Path = r'force_decouple_python/raw_data/'
savepath = r'force_decouple_python/decoupled/'
pathlist = os.listdir(r'force_decouple_python/raw_data')

b, a = signal.butter(1, 0.01, 'lowpass')  # 越小越光滑

def process_data_and_calculate_metrics(raw_data):
    # Load matrices and constants
    decouple_mix = pd.read_csv('force_decouple_python/m8301c_use/decoupled.csv', header=None)
    decouple_matrix = np.array(decouple_mix, dtype=np.float64)

    gain_mix = pd.read_csv('force_decouple_python/m8301c_use/gain.csv')
    GAIN = np.array(gain_mix.iloc[0:6], dtype=np.float64)
    EXC = gain_mix.iloc[6].value

    # Constants for calculations
    thrust0 = -0.49
    lift0 = 0.387
    mz0 = 0.02
    c = 0.06
    s = 0.22
    U = 0.09

    # Filtering parameters
    b, a = signal.butter(1, 0.01, 'lowpass')

    # Process raw data
    h0_theta0 = np.array([raw_data[7], raw_data[0]])
    h0_theta0_filtered = signal.filtfilt(b, a, h0_theta0)

    # Normalize measurements
    hmin, hmax = np.amin(h0_theta0_filtered[0]), np.amax(h0_theta0_filtered[0])
    tmin, tmax = np.amin(h0_theta0_filtered[1]), np.amax(h0_theta0_filtered[1])
    h0_theta0_filtered[0] = (h0_theta0_filtered[0] - (hmax + hmin) / 2) / 1.25675 * 0.1
    h0_theta0_filtered[1] = (h0_theta0_filtered[1] - (tmax + tmin) / 2) / 5.0 * 2 * np.pi

    # Decouple forces
    decoupled_force = decouple_matrix @ ((raw_data[1:7] * 1000.) / (EXC * GAIN))
    filtered_force = signal.filtfilt(b, a, decoupled_force)

    # Calculate thrust, lift, and moments in body coordinates
    thrust = -filtered_force[0] * np.sin(h0_theta0_filtered[1]) + filtered_force[1] * np.cos(
        h0_theta0_filtered[1]) - thrust0
    lift = -filtered_force[0] * np.cos(h0_theta0_filtered[1]) - filtered_force[1] * np.sin(
        h0_theta0_filtered[1]) - lift0
    mz = filtered_force[5] - mz0

    Ct     = thrust/(0.5*1000*U*U*c*s)
    Cl     = lift/(0.5*1000*U*U*c*s)
    Cpt    = thrust * U

    dim    = h0_theta0.shape[1]
    vy2     = np.zeros(dim)
    wz2     = np.zeros(dim)
    vy2[1:dim-1] = (h0_theta0[0,2:] - h0_theta0[0,0:dim-2]) /0.004 # 2个时间步差分
    wz2[1:dim-1] = (h0_theta0[1,2:] - h0_theta0[1,0:dim-2]) /0.004

    #Cpin                 = (-lift * vy2 - mz * wz2)/ (0.5 * 1000 * c * s * U * U * U) ##
    Cp_heave             = -lift * vy2 / (0.5 * 1000 * c * s * U * U * U)  #
    Cp_pitch             =  - mz * wz2 / (0.5 * 1000 * c * s * U * U * U)  #
    Cpin                 = Cp_heave + Cp_pitch

    Ct_mean   = np.zeros(Ct.shape[0])+1.0e-8
    Cpin_mean = np.zeros(Ct.shape[0])+1.0e-8
    Eta       = np.zeros(Ct.shape[0])+1.0e-8
    Energy_eta       = np.zeros(Ct.shape[0])+1.0e-8

    num_period = 505
    alpha=0
    for i in range(Ct.shape[0]-num_period-1):
        Ct_mean[i]   = np.mean( Ct[i:i+num_period] )
        Cpin_mean[i] = np.mean( Cpin[i:i+num_period] )
        Eta[i]       = Ct_mean[i]/Cpin_mean[i]


        Energy_eta[i]=Cpin_mean[i] *c /(2*( np.amax(h0_theta0_filtered[0])  +c*np.sin(np.amax(h0_theta0_filtered[1]))))

        if (vy2[i] < 0):
            alp=-(( h0_theta0_filtered[1][i]-math.pi) -np.atan(vy2[i]/U))
        else :
            alp=((h0_theta0_filtered[1][i]-math.pi)-np.atan(vy2[i]/U ))
        alpha += alp

    #np.mean(alpha), np.mean(Energy_eta[i])
    return np.mean(Ct_mean), np.mean(Eta)

# Example use:
# Assume `raw_data` is an array with shape (8, N) where N is the number of data points
# result_data = your_data_acquisition_function()  # This should get your 8-channel data
# Ct_mean, Cl_mean = process_data_and_calculate_metrics(result_data)
# print("Average Thrust Coefficient (CT):", Ct_mean)
# print("Average Lift Coefficient (CL):", Cl_mean)

