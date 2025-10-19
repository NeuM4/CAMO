import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import cigp as GPR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

class fidelity_kernel(nn.Module):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the fidelity kernel.
        """
        # Extract fidelity indicators
        f_1 = x1[:, -1]
        f_2 = x2[:, -1]

        # Compute the covariance matrix using the base kernels
        cov_matrix = self.kernel1(x1[:, :-1], x2[:, :-1]) * self.kernel2(f_1, f_2)
        
        return cov_matrix
    
class GP_SE2(nn.Module):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel = fidelity_kernel(kernel1, kernel2)
        self.cigp = GPR(kernel=self.kernel, log_beta = 1.0)
    
    def forward(self, data_manager, x_test, fidelity_indicator = None, normal = False):

        if fidelity_indicator is not None:
            x_test = torch.cat([x_test.reshape(-1,x_test.shape[1]),(torch.tensor(fidelity_indicator)+1).reshape(-1,1)], dim = 1)
        x_train, y_train = data_manager.get_data(0, normal = normal)
        y_pred, cov_pred = self.cigp(x_train,y_train,x_test)

        # return the prediction
        return y_pred, cov_pred
    
def train_GPSE2(GPSE2model, data_manager,max_iter=1000,lr_init=1e-1, normal = False):

    x_train, y_train = data_manager.get_data(0, normal = normal)

    optimizer = torch.optim.Adam(GPSE2model.parameters(), lr=lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = GPSE2model.cigp.negative_log_likelihood(x_train, y_train)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()), end='\r')
    print(' ')

# demo 
if __name__ == "__main__":

    torch.manual_seed(1)

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
    y_train = torch.cat((y_low, y_high1, y_high2), 0)
    y_test = torch.sin(x_test)

    # x_low = torch.cat((x_low, 0.33*torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
    # x_high1 = torch.cat((x_high1, 0.67*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
    # x_high2 = torch.cat((x_high2, torch.ones(x_high2.shape[0]).reshape(-1,1)), 1)
    # x_test = torch.cat((x_test, torch.ones(x_test.shape[0]).reshape(-1,1)), 1)

    x_low = torch.cat((x_low, torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
    x_high1 = torch.cat((x_high1, 2*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
    x_high2 = torch.cat((x_high2, 3*torch.ones(x_high2.shape[0]).reshape(-1,1)), 1)
    x_test = torch.cat((x_test, 3*torch.ones(x_test.shape[0]).reshape(-1,1)), 1)

    x = torch.cat((x_low, x_high1, x_high2), 0)
    y = torch.cat((y_low, y_high1, y_high2), 0)

    # initial_data = [
    #     {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low, 'Y': y_low},
    #     {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1, 'Y': y_high1},
    #     {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2, 'Y': y_high2},
    # ]
    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x.double(), 'Y': y.double()},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_1 = kernel.SquaredExponentialKernel()
    kernel_2 = kernel.SquaredExponentialKernel()
    GPSE2 = GP_SE2(kernel1=kernel_1, kernel2=kernel_2)

    train_GPSE2(GPSE2, fidelity_manager, max_iter=200, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = GPSE2(fidelity_manager,x_test.double())
 
    plt.figure()
    plt.errorbar(x_test[:,0].flatten(), ypred[:,0].reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test[:,0].flatten(), ypred[:,0].detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred[:,0].detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test[:,0].flatten(), y_test[:,0], 'k+')
    # plt.show()
    plt.savefig('GPSE2.png')
    pass