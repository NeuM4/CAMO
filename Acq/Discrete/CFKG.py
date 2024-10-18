import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from FidelityFusion_Models import *
from FidelityFusion_Models.GP_DMF import *
MF_model_list = {'CMF_CAR': ContinuousAutoRegression_large, 'CMF_CAR_dkl': CMF_CAR_dkl, "GP": cigp,
                 'ResGP': ResGP, 'AR': AR}

class discrete_fidelity_knowledgement_gradient(torch.nn.Module):
    def __init__(self, fidelity_num, GP_model, cost, data_model, data_manager, model_name, xdim, 
                 search_range, seed, xnorm = None, ynorm = None):
        super(discrete_fidelity_knowledgement_gradient, self).__init__()

        self.GP_model_pre = GP_model
        self.data_model = data_model
        self.cost = cost
        self.seed = seed
        self.search_range = search_range
        self.data_manager = data_manager
        self.model_name = model_name
        self.x_dim = xdim
        self.fidelity_num = fidelity_num
        self.x_norm = xnorm
        self.y_norm = ynorm

    def negative_cfkg(self, x, s):

        xall = torch.rand(100, self.x_dim,dtype=torch.float64) * (self.search_range[1] - self.search_range[0]) + self.search_range[0]
        # mean_y, sigma_y = self.GP_model_pre(xall, self.total_fid_num)  # 预测最高精度
        # xall = self.data_manager.normalizelayer[self.GP_model_pre.fidelity_num-1].normalize_x(xall)
        with torch.no_grad():
            if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
                if self.x_norm != None:
                    xall1 = self.x_norm.normalize(xall)
                else:
                    xall1 = xall
                mu_pre, var_pre = self.GP_model_pre(self.data_manager, xall1, torch.ones(100).reshape(-1,1)*(self.fidelity_num-1), normal = False)
                if self.y_norm != None:
                    mu_pre = self.y_norm.denormalize(mu_pre)
            else:
                if self.x_norm != None:
                    xall1 = self.x_norm[-1].normalize(xall)
                else:
                    xall1 = xall
                mu_pre, var_pre = self.GP_model_pre(self.data_manager, xall1, normal = False)
                if self.y_norm != None:
                    mu_pre = self.y_norm[-1].denormalize(mu_pre)
        # mu_pre, var_pre = self.data_manager.normalizelayer[self.GP_model_pre.fidelity_num-1].denormalize(mu_pre, var_pre)
        max_pre = torch.max(mu_pre)
        
        x = x.reshape(1,-1)
        with torch.no_grad():
            x = x.double()
            if self.x_norm != None:
                try:
                    x1 = self.x_norm[s].normalize(x)
                except:
                    x1 = self.x_norm.normalize(x)
            else:
                x1 = x
            ypred, _ = self.GP_model_pre(self.data_manager, x1, s, normal = False)
        
        if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
            x1 = torch.cat((x1, torch.tensor([[s]], dtype=torch.float64)),dim = 1)
            self.data_manager.add_data(raw_fidelity_name = '0',fidelity_index= 0 , x = x1, y=ypred)
        else:
            self.data_manager.add_data(raw_fidelity_name=str(s), fidelity_index=s, x = x1, y=ypred)
        ## can it be imporved?
        print('train new GP model')
        kernel_init1 = kernel.SquaredExponentialKernel(length_scale=1., signal_variance=1.)
        kernel_init2 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(self.fidelity_num)]
        if self.model_name == 'ResGP':
            GP_model_new = MF_model_list['ResGP'](fidelity_num = self.fidelity_num, kernel_list = kernel_init2, if_nonsubset = True)
            train_ResGP(GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'AR':
            GP_model_new = MF_model_list['AR'](fidelity_num = self.fidelity_num, kernel_list = kernel_init2, if_nonsubset = True)
            train_AR(GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        # elif self.model_name == 'CAR':
        #     train_CAR(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        # elif self.model_name == 'DMF_CAR':
        #     train_DMFCAR(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        # elif self.model_name == 'DMF_CAR_dkl':
        #     train_DMFCAR_dkl(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        
        elif self.model_name == 'CMF_CAR':
            GP_model_new = MF_model_list['CMF_CAR'](kernel_x=kernel_init1)
            train_CAR_large(GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'CMF_CAR_dkl':
            GP_model_new = MF_model_list['CMF_CAR_dkl'](input_dim=x.shape[1], kernel_x=kernel_init1)
            train_CMFCAR_dkl(GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'GP':
            GP_model_new = MF_model_list['GP'](kernel=kernel_init1, log_beta=1.0)
            train_GP(GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        
        with torch.no_grad():
            if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
                if self.x_norm != None:
                    xall2 = self.x_norm.normalize(xall)
                else:
                    xall2 = xall
                mu, var = GP_model_new(self.data_manager, xall2, torch.ones(100).reshape(-1,1), normal=False)
                if self.y_norm != None:
                    mu = self.y_norm.denormalize(mu)
            else:
                if self.x_norm != None:
                    xall2 = self.x_norm[-1].normalize(xall)
                else:
                    xall2 = xall
                mu, var = GP_model_new(self.data_manager, xall2, normal=False)
                if self.y_norm != None:
                    mu = self.y_norm[-1].denormalize(mu)
        # mu, _ = self.data_manager.normalizelayer[self.GP_model_new.fidelity_num-1].denormalize(mu, var)
        
        if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
            self.data_manager.data_dict['0']['X'] = self.data_manager.data_dict['0']['X'][:-1]
            self.data_manager.data_dict['0']['Y'] = self.data_manager.data_dict['0']['Y'][:-1]
        else:
            self.data_manager.data_dict[str(s)]['X'] = self.data_manager.data_dict[str(s)]['X'][:-1]
            self.data_manager.data_dict[str(s)]['Y'] = self.data_manager.data_dict[str(s)]['Y'][:-1]
        max_mu = torch.max(mu)
        c = self.cost.compute_cost(s / self.fidelity_num)
        cfkg = (max_mu - max_pre) / c

        return cfkg

    def compute_next(self):
        # torch.manual_seed(self.seed)
        N = 3
        sample_x = []
        for i in range(self.x_dim):
            torch.manual_seed(self.seed + 86 + i)
            sample_x.append(torch.rand(N, 1) * (self.search_range[1] - self.search_range[0]) + self.search_range[0])
        sample_x = torch.cat(sample_x, axis=1)

        # s = torch.ones(N) + 1
        torch.manual_seed(self.seed + 86 + 37)
        s = torch.randint(0, self.fidelity_num, (N,), dtype=torch.float)
        
        for i in range(N):
            cfkg = self.negative_cfkg(sample_x[i], int(s[i]))
            if i == 0:
                max_cfkg = cfkg
                new_x = sample_x[i].reshape(-1,1)
                new_s = int(s[i])
            else:
                if cfkg > max_cfkg:
                    max_cfkg = cfkg
                    new_x = sample_x[i].reshape(-1,1)
                    new_s = int(s[i])

        return new_x.reshape(1,-1), new_s

