import pandas as pd
from math import cos, sin, tan, radians
from math import atan
from math import pi
from math import pow
from math import sqrt
# 读取上传的CSV文件
file_path = 'Database/centroids.csv'
naca_params_df = pd.read_csv(file_path, header=None)


import numpy as np

# 重新定义反归一化的边界
UPB = np.array([9, 9, 35])
LOWB = np.array([-9, 0, 10])

# 进行反归一化
denormalized_params = LOWB + (UPB - LOWB) * naca_params_df

# 对反归一化后的值取绝对值
denormalized_params_abs = denormalized_params
#denormalized_params_abs[1][5]=1.5
print(denormalized_params_abs)

# 定义所需的变量
numpoints = 1000 # 数据点的数量
c = 60  # 翼型的弦长
class Display(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.h = []
        self.label = []
        self.fig, self.ax = self.plt.subplots()
        self.plt.axis('equal')
        self.plt.xlabel('x')
        self.plt.ylabel('y')
        self.ax.grid(True)
    def plot(self, X, Y,label=''):
        h, = self.plt.plot(X, Y, '-', linewidth = 1)
        self.h.append(h)
        self.label.append(label)
    def show(self):
        self.plt.axis((-0.1,1.1)+self.plt.axis()[2:])
        self.ax.legend(self.h, self.label)
        self.plt.show()
def linspace(start,stop,np):
    """
    Emulate Matlab linspace
    """
    return [start+(stop-start)*i/(np-1) for i in range(np)]
def naca4(number, n, finite_TE = False, half_cosine_spacing = False):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
    """

    m = float(number[0])/100.0
    p = float(number[1])/10.0
    t = float(number[2])/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_TE:
        a4 = -0.1015 # For finite thick TE
    else:
        a4 = -0.1036 # For zero thick TE

    if half_cosine_spacing:
        beta = linspace(0.0,pi,n+1)
        x = [(0.5*(1.0-cos(xx))) for xx in beta]  # Half cosine based spacing
    else:
        x = linspace(0.0,1.0,n+1)

    yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-xx for xx in yt]

        xc = xc1 + xc2
        zc = [0]*len(xc)
    else:
        yc1 = [m/pow(p,2)*xx*(2*p-xx) for xx in xc1]
        yc2 = [m/pow(1-p,2)*(1-2*p+xx)*(1-xx) for xx in xc2]
        zc = yc1 + yc2

        dyc1_dx = [m/pow(p,2)*(2*p-2*xx) for xx in xc1]
        dyc2_dx = [m/pow(1-p,2)*(2*p-2*xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx

        theta = [atan(xx) for xx in dyc_dx]

        xu = [xx - yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yu = [xx + yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

        xl = [xx + yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yl = [xx - yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X,Z

def generate_naca_data(m, p, t, i):
    # 重新定义x坐标的范围为0到1
    x = np.linspace(0, 1, numpoints)
    # 计算基本翼型形状
    y = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    # 计算弯度部分
    k = 0.1 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    camber = m * k * (1 - np.cos(np.pi * p * (x - 0.5))) / 2
    
    # 计算上表面和下表面的y坐标
    yupper = y + camber
    ylower = y - camber

    # 调整y坐标，加上i*0.5
    yupper += i * 0.5
    ylower += i * 0.5
    file_name = f'naca_airfoil_{i+1}.txt'
    # 将数据保存到.txt文件，格式为：x坐标, 上表面y坐标, 下表面y坐标
    with open(file_name, 'w') as file:
        for i in range(numpoints):
            file.write(f"{x[i]*c} {yupper[i]*c} {0}\n")
        for i in reversed(range(numpoints)):     
            file.write(f"{x[i]*c} {ylower[i]*c} {0}\n")
    return x,yupper


# 生成并保存八个CSV文件的循环
for i, row in denormalized_params_abs.iterrows():
    x,y=naca4(row,numpoints)
    p=Display()
    p.plot(x,y)
    file_name = f'naca_airfoil_{i+1}.txt'
    
    # 将数据保存到.txt文件，格式为：x坐标, 上表面y坐标, 下表面y坐标
    with open(file_name, 'w') as file:
        for j in range(len(x)):
            y[j] = y[j]*c+i * c
            file.write(f"{x[j]*c} {y[j]} {0}\n")
        # for i in reversed(range(numpoints)):     
        #     file.write(f"{x[i]*c} {y[i]*c} {0}\n")
p.show()


i+=1
c=90
num=(0,0,12)
x,y=naca4(num,numpoints)
p=Display()
p.plot(x,y)
file_name = f'naca_airfoil_{i+1}.txt'
with open(file_name, 'w') as file:
    for j in range(len(x)):
        y[j] = y[j]*c+i * c
        file.write(f"{x[j]*c} {y[j]} {0}\n")

i+=1
c=120
x,y=naca4(num,numpoints)
p=Display()
p.plot(x,y)
file_name = f'naca_airfoil_{i+1}.txt'
with open(file_name, 'w') as file:
    for j in range(len(x)):
        y[j] = y[j]*c+i * c
        file.write(f"{x[j]*c} {y[j]} {0}\n")
p.show()