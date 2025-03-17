import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 从真实数据集加载数据（saveY）
saveY = np.loadtxt('saveY_gpytorch_multifidelity_multitask.csv', delimiter=',')  # 真实数据路径
# 假设 saveY 的列分别是 CT, CL, eta
ct_real = saveY[:, 0]
cl_real = saveY[:, 1]
eta_real = saveY[:, 2]

# 从 'pareto_front.csv' 加载预测前沿数据
pareto_df = pd.read_csv('pareto_front.csv')  # 预测前沿数据路径
pareto_F = pareto_df.iloc[:, -3:].values  # 只取目标值列 (CT, CL, eta)

# 定义函数来计算非支配解
def is_dominated(x, population):
    """检查解 x 是否被种群中的解支配"""
    for p in population:
        # 如果在所有目标上都不比 p 差，说明 x 被 p 支配
        if np.all(p <= x) and np.any(p < x):
            return True
    return False

def get_pareto_front(data):
    """返回非支配解（帕累托前沿）"""
    pareto_front = []
    for i in range(data.shape[0]):
        if not is_dominated(data[i], data[pareto_front]):
            pareto_front.append(i)
    return data[pareto_front]

# 获取真实前沿和预测前沿的非支配解
real_data = np.column_stack([ct_real, cl_real, eta_real])
pred_data = pareto_F

# 过滤掉 eta > 1 的点
real_data_filtered = real_data[real_data[:, 2] <= 1]
pred_data_filtered = pred_data[pred_data[:, 2] <= 1]

# 获取过滤后的真实前沿和预测前沿的非支配解
real_pareto_front = get_pareto_front(real_data_filtered)
pred_pareto_front = get_pareto_front(pred_data_filtered)

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制真实前沿的非支配解 (蓝色)
ax.scatter(real_pareto_front[:, 0], real_pareto_front[:, 1], real_pareto_front[:, 2], c="blue", marker="o", label="数据集前沿", s=20)

# 绘制预测前沿的非支配解 (红色)
ax.scatter(pred_pareto_front[:, 0], pred_pareto_front[:, 1], pred_pareto_front[:, 2], c="red", marker="o", label="预测前沿", s=20)

# 设置坐标轴标签，使用 LaTeX 格式
ax.set_xlabel(r'$C_T$', fontsize=16, fontfamily='Times New Roman')
ax.set_ylabel(r'$C_L$', fontsize=16, fontfamily='Times New Roman')
ax.set_zlabel(r'$\eta$', fontsize=16, fontfamily='Times New Roman')

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=14)

# 设置全局字体
plt.rc('font', family='SimHei')

# 调整子图位置，以便显示标签
plt.subplots_adjust(left=0.15, bottom=0.15)

# 添加图例
ax.legend()

# 保存图形为高分辨率的 PDF 文件
plt.savefig("pareto_comparison_3d.pdf", dpi=300)

# 显示图形
plt.show()
