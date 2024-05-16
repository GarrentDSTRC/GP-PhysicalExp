import numpy as np
import os
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

controlFre = 500
c=0.06
U=0.06
mode="experiment"
def generate_waveform( X, folder_name,mode="CFD"):
    # 创建文件夹（如果不存在）
    St, amplitude2, amplitude, phase_difference, alpha, alpha2=X
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if mode=="CFD":
        f=St
    else:
        f = X[0] * U / c
    T = 1 / f
    #print(T)
    points = T * controlFre


    # y(t)定义
    def y(t):
        return np.sin(t)

    # φ(t')定义
    def phi(t):
        return t + alpha * np.sin(t) ** 2

    # 使用数值方法找到φ^-1(t')的对应t值
    def phi_inverse(phi_prime, t_values):
        # 计算每一个t值对应的φ(t)和给定的t'之间的差异
        diffs = phi(t_values) - phi_prime
        # 找到差异最小的t值
        idx = np.argmin(np.abs(diffs))
        return t_values[idx]

    # 计算z(t')
    def z(phi_prime, t_values):
        t = phi_inverse(phi_prime, t_values)
        return y(t)

    # 主程序

    t_values = np.linspace(0, 2 * np.pi, int(points))
    phi_values = phi(t_values)

    # 计算z(t')的值
    z_values = [(amplitude * np.pi / 180) * z(phi_prime, t_values) for phi_prime in phi_values]
    # 使用线性插值生成均匀的时间点
    phi_uniform = t_values
    f_interp = interp1d(phi_values, z_values)
    z_uniform = f_interp(phi_uniform)



    alpha=alpha2
    # 生成第二个波形
    phi_values2 = phi(t_values)
    z_values2 = [(amplitude2) * z(phi_prime, t_values) for phi_prime in phi_values2]

    # 使用线性插值生成均匀的时间点
    phi_uniform2 =t_values
    f_interp2 = interp1d(phi_values2, z_values2)
    z_uniform2 = f_interp2(phi_uniform2)

    num_rolls = int(-phase_difference/360 / T * len(z_uniform))
    z_uniform = np.roll(z_uniform, num_rolls)
   # 保存第一个波形到文件

    with open(os.path.join(folder_name, "control.txt"), "w") as f:
        for value in z_uniform:
                f.write(str(value) + "\n")
    # 保存第二个波形到文件
    if mode == "CFD":
        with open(os.path.join(folder_name, "control2.txt"), "w") as f2:
            for value in z_uniform2:
                f2.write(str(value) + "\n")
        with open(os.path.join(folder_name, "control.txt"), "w") as f:
            for value in z_uniform:
                f.write(str(value) + "\n")

    else:
        with open(os.path.join(folder_name, "control2.txt"), "w") as f2:
            for value in z_uniform2:
                f2.write(str(value*100*4.5) + "\n")
        with open(os.path.join(folder_name, "control.txt"), "w") as f:
            for value in z_uniform:
                f.write(str(value*180/np.pi*3) + "\n")


    # 绘制第一个波形
    plt.plot(phi_uniform, z_uniform, color="green", label="Pitching")
    # 绘制第二个波形
    plt.plot(phi_uniform2, z_uniform2, color="blue", label="Heaving")
    plt.xlabel("φ")
    plt.ylabel("z(φ)")
    plt.legend()
    #plt.show()
    plt.savefig("waveform.png")

    return f"Waveforms saved to {folder_name}/control.txt and {folder_name}/control2.txt"

UPB=[1.0, 0.6, 40, 180, 1,1,1000]
LOWB=[0.4, 0.1, 5, -180, -1,-1,100]
import torch
class Normalizer:
    def __init__(self, low_bound=LOWB, up_bound=UPB):
        self.low_bound = torch.tensor(low_bound, dtype=torch.float32)
        self.up_bound = torch.tensor(up_bound, dtype=torch.float32)

    def normalize(self, x):
        x=torch.as_tensor(x)
        return (x - self.low_bound) / (self.up_bound - self.low_bound)

    def denormalize(self, norm_x):
        norm_x = torch.as_tensor(norm_x)
        return norm_x * (self.up_bound - self.low_bound) + self.low_bound

if __name__=="__main__":
    norm=Normalizer()
    #x=[9.90E-01,	9.30E-01,	9.00E-01,	3.50E-01,	3.20E-01,	9.50E-01,	3.10E-01] #0015
    #x=[9.61E-01,	7.03E-02,	4.45E-01,	1.17E-01,	3.67E-01,	3.36E-01,	9.30E-01]#RANDOM ANGLE*-1
    #x=[9.80E-01,	9.80E-01,	8.80E-01,	7.90E-01,	8.60E-01,	0.00E+00,0.5]#0024
    #x=[0.99	,0.87,0.88,0.85,	0.95	,0.98,0.5]#6129 RANDOM2 ANGLE*-1
    #x=[5.30E-01,	2.10E-01,	8.50E-01,	1.00E+00,	9.80E-01,	4.30E-01,	9.10E-01]#采集
    #0.80	0.87	0.60	0.65	0.43	0.01 #6129 2
    X=[1,	0.6, 40,	0,	0.95,	9.50E-01,	3.10E-01]
    #X=norm.denormalize(x).tolist()



    last_col = X[-1]  # Extract the last column
    j=0
    np.savetxt(r'.\MMGP_OL%d\dataX.txt' % (j % 8), np.array([[0, 0, 0, 0, 0, 0, 20, 12000]]),
                           delimiter=',', fmt='%d')
    generate_waveform(X[0:-1],"MMGP_OL%d"% (j % 8),mode=mode)
    # Test
