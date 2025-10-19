import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def derive_forward(length_scale, x1, x2):

    log_length_scales = torch.log(torch.tensor([length_scale]))
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

t_all = torch.linspace(0, 10, 11).reshape(-1, 1)
x_all = torch.rand(t_all.shape[0], 1) * 20
x_test = torch.cat((x_all, t_all), 1)

record = []
length_scale = torch.cat((torch.linspace(0.5, 6, 30), torch.linspace(6, 10, 20)), 0)

for len in length_scale:
    Sigma_derive = derive_forward(len, x_test, x_test)
    record.append(Sigma_derive.diag().detach().numpy())

rec = np.asarray(record)

X, Y = np.meshgrid(length_scale, t_all)
Z = rec.T

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf_1 = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', label = 'ARD')

ax.set_xlabel('Length_scales')
ax.set_ylabel('t')
ax.set_zlabel('cov(t, t)')

fig.colorbar(surf_1, ax=ax, shrink=0.5, aspect=5)

# plt.legend()
plt.show()

# plt.savefig('Kernel_compare_1.png') 


