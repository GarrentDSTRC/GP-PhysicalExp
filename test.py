import numpy as np
import time


dir0 = r'D:\PycharmProjects\EXP_daq_20240311_force measurement\exp20230330\推力\0024\非对称'
dir_p = dir0 + '\control.txt'
dir_h = dir0 + '\control2.txt'
datap = np.loadtxt(dir_p)
datah = np.loadtxt(dir_h)

length = datap.shape[0]
print(datap.shape)
print(datah.shape)

t0 = time.time()

for i in range(4*1000):
    t1= time.time() - t0
    idx = int(1000*t1) % length
    heave = datah[idx] * 1000
    pitch = datap[idx] * 180.0/np.pi
    time.sleep(0.005)
    #print(t1, heave, pitch)
    print(np.max(datah*1000))
