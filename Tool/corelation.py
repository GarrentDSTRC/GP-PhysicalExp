import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import io
import sys

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据文件
df = pd.read_csv('Tool/train_y.csv', 
                header=None, 
                names=['Column1', 'Column2'], 
                sep=',')

# 数据质量检查
print("缺失值检查:")
print(df.isnull().sum())

# 计算统计指标
corr, p_value = stats.pearsonr(df['Column1'], df['Column2'])
print(f'相关系数: {corr:.3f}')
print(f'P值: {p_value:.4f}')
print('相关性显著' if p_value < 0.05 else '相关性不显著')

# 配置字体和输出参数
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 300
})

# 创建图形对象
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点图
ax.scatter(df['Column1'], df['Column2'], 
          edgecolors='b', 
          alpha=0.7)

# 设置坐标轴标签
ax.set_xlabel(r'$\eta$', fontsize=18)
ax.set_ylabel(r'$\beta$', fontsize=18)

# 调整页面布局
plt.tight_layout()

# 保存为PDF文件（矢量图格式）
plt.savefig('corela_aym.pdf', format='pdf', bbox_inches='tight')

# 清理内存
plt.close()