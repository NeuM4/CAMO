import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch.nn as nn
import torch
from FidelityFusion_Models.GP_DMF import *
from FidelityFusion_Models.CMF_CAR import *

MF_model_list = {'CMF_CAR': ContinuousAutoRegression_large, "GP": cigp}

class continuous_fidelity_knowledgement_gradient(nn.Module):
    def __init__(self, x_dimension, posterior_function, data_model, model_cost, data_manager, seed, search_range,norm,model_name):
        super(continuous_fidelity_knowledgement_gradient, self).__init__()

        self.x_dimension = x_dimension
        self.pre_func = posterior_function
        self.data_model = data_model
        self.model_cost = model_cost
        self.data_manager = data_manager
        self.seed = seed
        self.model_name = model_name
        self.x_norm = norm[0]
        self.y_norm = norm[1]
        self.search_range = search_range

    def negative_cfkg(self, x, s):
        
        _, x_te = self.data_model.find_max_value_in_range()
        xte = x_te[:100].double()

        with torch.no_grad():
            xte = self.x_norm.normalize(xte)
            mean_y, _ = self.pre_func(self.data_manager, xte, torch.ones(100).reshape(-1,1))  # 预测最高精度
            mean_y = self.y_norm.denormalize(mean_y)
        max_mean_y = torch.max(mean_y)
        
        with torch.no_grad():
            x1 = self.x_norm.normalize(x)
            y, var = self.pre_func(self.data_manager, x1, s+1)
            y = self.y_norm.denormalize(y)
            
        x = torch.cat((x, s), dim=1)
        x1 = self.x_norm.normalize(x)
        self.data_manager.add_data(raw_fidelity_name = '0',fidelity_index= 0 , x=x1, y=y)

        print('train new GP model')
        kernel_init = kernel.SquaredExponentialKernel(length_scale=1., signal_variance=1.)
        if self.model_name == "CMF_CAR":
            GP_new = MF_model_list['CMF_CAR'](kernel_x = kernel_init)
            train_CAR_large(GP_new, self.data_manager, max_iter=100, lr_init=1e-2)
        elif self.model_name == "GP":
            GP_new = MF_model_list['GP'](kernel = kernel_init, log_beta=1.0)
            train_GP(GP_new, self.data_manager, max_iter=100, lr_init=1e-2)
        
        with torch.no_grad():
            xte = self.x_norm.normalize(xte)
            mu, _ = GP_new.forward(self.data_manager, xte, torch.ones(100).reshape(-1,1)) 
            mu = self.y_norm.denormalize(mu)
        
        self.data_manager.data_dict['0']['X'] = self.data_manager.data_dict['0']['X'][:-1]
        self.data_manager.data_dict['0']['Y'] = self.data_manager.data_dict['0']['Y'][:-1]
        max_mu = torch.max(mu)
        c = self.model_cost.compute_cost(s)
        cfkg = (max_mu.item() - max_mean_y.item()) / c

        return cfkg

    def compute_next(self):

        N = 3 # adjustable
        tt = []
        for i in range(self.x_dimension):
            torch.manual_seed(self.seed + 86 + i)
            tt.append(torch.rand(N, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0])
        tt = torch.cat(tt, dim=1)
        tt = tt.double()

        torch.manual_seed(self.seed + 86 + 37)
        ts = torch.rand(N, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + self.search_range[-1][0]

        max_cfkg = float("-inf")
        for i in range(N):
            cfkg = self.negative_cfkg(tt[i].reshape(1, tt.shape[1]), ts[i].reshape(1, 1))
            if cfkg > max_cfkg:
                max_cfkg = cfkg
                new_x = tt[i].reshape(1, tt.shape[1])
                new_s = ts[i].reshape(1, 1)

        return new_x, new_s

