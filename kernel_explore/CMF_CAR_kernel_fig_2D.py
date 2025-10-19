import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ard_forward(x1, x2):
        log_length_scales = torch.zeros(1)

        X1 = x1[:, 0].reshape(-1, 1)
        X2 = x2[:, 0].reshape(-1, 1)
        fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        scaled_x1 = fidelity_indicator_1 / log_length_scales.exp()
        scaled_x2 = fidelity_indicator_2 / log_length_scales.exp()
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2

        return torch.exp(-0.5 * sqdist)



def derive_forward(x1, x2):

    log_length_scales = torch.tensor([0.5])
    b = torch.tensor([1])
    v = log_length_scales.exp() * b * 0.5

    def h(t, t_1):
        tem_1 = (v**2).exp() / (2*b)
        tem_2 = (-b * t).exp() 
        tem_3 = (b * t_1).exp() * (torch.erf((t-t_1)/log_length_scales.exp() - v) + torch.erf((t_1)/log_length_scales.exp() + v))
        tem_4 = (-b * t_1).exp() * (torch.erf((t/log_length_scales.exp()) - v) + torch.erf(v))

        return tem_1 * tem_2 * (tem_3 - tem_4)
        
    fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
    fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

    tem = [fidelity_indicator_1 for i in range(fidelity_indicator_2.size(0))]
    T1 = torch.cat(tem, dim=1)
    tem = [fidelity_indicator_2 for i in range(fidelity_indicator_1.size(0))]
    T2 = torch.cat(tem, dim=1).T

    h_part_1 = h(T1, T2)
    h_part_2 = h(T2, T1)

    final_part = 0.5 * torch.sqrt(torch.tensor(torch.pi)) * log_length_scales.exp() * (h_part_1 + h_part_2)


    return final_part

t_all = torch.linspace(0, 1, 50).reshape(-1, 1) * 2 + 1
x_all = torch.rand(50, 1) * 20
x_test = torch.cat((x_all, t_all), 1)

# t_all_2 = torch.linspace(0, 1, 10).reshape(-1, 1)
# x_all_2 = torch.rand(10, 1) * 20
# x_test_2 = torch.cat((x_all_2, t_all_2), 1)

Sigma_ard = ard_forward(x_test, x_test)
Sigma_derive = derive_forward(x_test, x_test)

import numpy as np

# 生成曲面数据
X, Y = np.meshgrid(t_all, t_all)
Z_ard = Sigma_ard.detach().numpy()
Z_derive = Sigma_derive.detach().numpy()

# 绘制曲面图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf_1 = ax.plot_surface(X, Y, Z_ard, cmap='viridis', edgecolor='none', label = 'ARD')
surf_2 = ax.plot_surface(X, Y, Z_derive, cmap='plasma', edgecolor='none', label = 'Derive')
fig.colorbar(surf_1, ax=ax, shrink=0.5, aspect=5)
 
plt.legend()
plt.show()
# plt.savefig('Kernel_compare.png') 
