import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class AR_union(nn.Module):
    
    def __init__ (self, fidelity_num, kernel_list, rho_init = 1.0):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.kernel_list = nn.ModuleList(kernel_list)
        self.rho_list = []
        for _ in range(self.fidelity_num - 1):
            self.rho_list.append(torch.nn.Parameter(torch.tensor(rho_init)))
        self.rho_list = torch.nn.ParameterList(self.rho_list)
    
    def forward(self, data_manager, x_test, fidelity_indicator = None, normal = False):
        
        if fidelity_indicator is not None:
            x_test = torch.cat([x_test.reshape(-1,x_test.shape[1]),(torch.tensor(fidelity_indicator)+1).reshape(-1,1)], dim = 1)
        
        to_fidelity = x_test[0,-1].int()
        
        x = []
        y = []
        x_num = 0
        for f in range(self.fidelity_num):
            x_f, y_f = data_manager.get_data(f, normal= normal)
            x_num += x_f.shape[0]
            x.append(x_f)
            y.append(y_f)
        x_train = x
        y_train = torch.cat(y, dim=0)
        
        Sigma = self.cal_Union_sigma(x_num, x_train)
        kx =  self.cal_union_kx(x_num, x_test.shape[0], x_train, x_test, to_fidelity)
        L = torch.linalg.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)
        mean = kx.t() @ torch.cholesky_solve(y_train, L)
        var = self.cal_Union_sigma(x_test.shape[0], x_test) - LinvKx.t() @ LinvKx
        # var = var + self.log_beta.exp().pow(-1)
        
        return mean, var
            
    
    def negative_log_likelihood(self, x_train, y_train):
        
        x_num = 0
        for i in range(self.fidelity_num):
            x_num += x_train[i].shape[0]
        y_num, y_dimension = x_num , y_train.shape[1]
        
        Sigma = self.cal_Union_sigma(x_num, x_train)   
            
        L = torch.linalg.cholesky(Sigma)
        Gamma,_ = torch.triangular_solve(y_train, L, upper = False)
        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll
    
    def cal_Union_sigma(self, x_num, x_train):
        
        Sigma = torch.zeros(x_num, x_num)
        cur_row = 0
        cur_col = 0
        for f_r in range(self.fidelity_num):
            for f_c in range(f_r, self.fidelity_num):
                x_f = x_train[f_r]
                x_c = x_train[f_c]
                xf_num = x_f.shape[0]
                xc_num = x_c.shape[0]
                
                if f_r == f_c and f_r == 0:
                    Sigma[cur_row:cur_row+xc_num, cur_col:cur_col+xf_num] = self.kernel_list[0](x_f, x_c)
                elif f_r == f_c:
                    Sigma[cur_row:cur_row+xc_num, cur_col:cur_col+xf_num] = self.kernel_list[0](x_f, x_c)
                    for i in range(f_r):
                        Sigma[cur_row:cur_row+xc_num, cur_col:cur_col+xf_num] += self.rho_list[i]**2 * self.kernel_list[i+1](x_f, x_c)
                else:
                    # continue
                    Sigma[cur_row:cur_row+xc_num, cur_col:cur_col+xf_num] = self.rho_list[f_c - 1] * self.kernel_list[f_c](x_f, x_c)
                    Sigma[cur_col:cur_col+xf_num, cur_row:cur_row+xc_num] = Sigma[cur_row:cur_row+xc_num, cur_col:cur_col+xf_num].T
                cur_col += xc_num
            cur_row += x_train[f_r].shape[0]
            cur_col = cur_row    
            
        return Sigma
    
    def cal_union_kx(self, x1_num, x2_num, x_train, x_test, to_fidelity):
        kx = torch.zeros(x1_num, x2_num)
        cur_row = 0
        rho_num = to_fidelity
        for f_r in range(self.fidelity_num):
            x_f = x_train[f_r]
            xf_num = x_f.shape[0]
            if f_r == to_fidelity:
                kx[cur_row:cur_row+xf_num,0:x2_num] = self.kernel_list[0](x_f, x_test)
                for i in range(f_r):
                    kx[cur_row:cur_row+xf_num,0:x2_num] += self.rho_list[i]**2 * self.kernel_list[i+1](x_f, x_test)
            elif f_r < to_fidelity:
                kx[cur_row:cur_row+xf_num,0:x2_num] = self.rho_list[to_fidelity-1] * self.kernel_list[to_fidelity-1](x_f, x_test)
            else:
                kx[cur_row:cur_row+xf_num,0:x2_num] = self.rho_list[rho_num] * self.kernel_list[rho_num](x_test, x_f)
                rho_num += 1
            cur_row += xf_num
        return kx   
            
def train_AR_union(GPmodel, data_manager, max_iter=100, lr_init=1e-1, normal = False):
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr = lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        
        x = []
        y = []
        for f in range(GPmodel.fidelity_num):
            x_f, y_f = data_manager.get_data(f, normal = normal)
            x.append(x_f)
            y.append(y_f)
        y = torch.cat(y, dim=0)
        
        loss = GPmodel.negative_log_likelihood(x, y)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()), end='\r')
    print('',end='\n')
    
if __name__ == "__main__":
    
    torch.manual_seed(1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    xlow_indices = torch.sort(xlow_indices).values
    x_low = x_all[xlow_indices]
    xhigh1_indices = torch.randperm(500)[:300]
    xhigh1_indices = torch.sort(xhigh1_indices).values
    x_high1 = x_all[xhigh1_indices]
    xhigh2_indices = torch.randperm(500)[:250]
    xhigh2_indices = torch.sort(xhigh2_indices).values
    x_high2 = x_all[xhigh2_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) - 0.5 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
    y_high1 = torch.sin(x_high1) - 0.3 * torch.sin(2 * x_high1) + torch.rand(300, 1) * 0.1 - 0.05
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 - 0.05
    y_test = torch.sin(x_test)
    
    x_low = torch.cat((x_low, torch.zeros(x_low.shape[0], 1)), dim=1)
    x_high1 = torch.cat((x_high1, torch.ones(x_high1.shape[0], 1)), dim=1)
    x_high2 = torch.cat((x_high2, torch.full((x_high2.shape[0], 1), 2)), dim=1)
    x_test = torch.cat((x_test, torch.full((x_test.shape[0], 1), 2)), dim=1)
    
    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low[:4].to(device), 'Y': y_low[:4].to(device)},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1[:4].to(device), 'Y': y_high1[:4].to(device)},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2[:4].to(device), 'Y': y_high2[:4].to(device)},
    ]
    fidelity_num = len(initial_data)

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    myAR = AR_union(fidelity_num = fidelity_num, kernel_list = kernel_list, rho_init=1.0).to(device)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_AR_union(myAR, fidelity_manager, max_iter=200, lr_init=1e-3)

    with torch.no_grad():
        # x_test = fidelity_manager.normalizelayer[myAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myAR(fidelity_manager,x_test)
        # ypred, ypred_var = fidelity_manager.normalizelayer[myAR.fidelity_num-1].denormalize(ypred, ypred_var)

    plt.figure()
    plt.errorbar(x_test[:,0].flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test[:,0].flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test[:,0].flatten(), y_test, 'k+')
    plt.savefig('AR_union.png')
    # plt.show()
    