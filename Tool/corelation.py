import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import io
import sys
# 设置标准输出的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 读取CSV文件
df = pd.read_csv('train_y.csv', header=None, names=['Column1', 'Column2'], sep=',')

# 检查缺失值
print("缺失值检查:")
print(df.isnull().sum())

# 计算皮尔逊相关系数和p值
corr, p_value = stats.pearsonr(df['Column1'], df['Column2'])

# 输出结果
print('相关系数为:', corr)
print('对应的p值为:', p_value)

# 判断显著性
if p_value < 0.05:
    print('相关性在0.05水平上显著。')
else:
    print('相关性不显著。')

plt.scatter(df['Column1'], df['Column2'])
plt.xlabel(r'$\eta$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
#plt.title('Scatter Plot: Column1 vs Column2', fontsize=18)
plt.show()