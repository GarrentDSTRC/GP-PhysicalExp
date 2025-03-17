import torch

#预测值
A = torch.tensor([[-2.1204,  0.2320, -0.2975],
        [ 4.6143,  0.8636,    float('-inf')],
        [ 1.7967,  2.3266, -6.4257],
        [-2.2641,  0.2772, -0.2197],
        [ 1.3727,  0.1817, -4.9624],
        [-0.4345,  0.9692, -1.8248]], dtype=torch.float64)
#真实值
B = torch.tensor([
    [-1.25071,  0.36303484,  0.4612234],
    [ 6.761007786,  4.90068388,  float('-inf')],
    [ 1.220884204,  6.062333584,  float('-inf')],
    [-1.6765177,  0.2595094,  0.41716385],
    [ 1.9103503,  -0.4903468, -12.222097],
    [-0.41891593,  1.4779946, -4.185682]
], dtype=torch.float64)

# # 将nan替换为10
# A[torch.isnan(A)] = 10
# B[torch.isnan(B)] = 10

# 计算arctan
A1 = 3*torch.tanh(A/3)
B1 =3* torch.tanh(B/3)

# A[torch.isnan(A)] = -1
# B[torch.isnan(B)] = -1
# 计算MAE
MAE = torch.mean(torch.abs(A1 - B1))
print(MAE)
