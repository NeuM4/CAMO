import torch
import matplotlib.pyplot as plt
import numpy as np

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

t_all = torch.linspace(0,10, 11).reshape(-1, 1)
x_all = torch.rand(t_all.shape[0], 1) * 20
x_test = torch.cat((x_all, t_all), 1)

record = []
length_scale = torch.cat((torch.linspace(0.5, 6, 30), torch.linspace(6, 10, 20)), 0)

for len in length_scale:
    Sigma_derive = derive_forward(len, x_test, x_test)
    record.append(Sigma_derive.diag().detach().numpy())

rec = np.asarray(record)

plt.figure()
color_dic = ['#8A2BE2', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000']

for i in range(t_all.shape[0]):
    plt.plot(length_scale.numpy().flatten(), rec[:, i].flatten(), color = color_dic[i], ls = 'dashed', label = 't =' + str(t_all[i].item()))

plt.xlabel("Length_scale")
plt.ylabel("Cov(t, t)")
plt.legend()
plt.show()
# plt.savefig('Kernel_compare_1.png') 


