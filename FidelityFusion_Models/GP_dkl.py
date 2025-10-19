import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from matplotlib import pyplot as plt
import time as time
from FidelityFusion_Models.MF_data import MultiFidelityDataManager

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class cigp_dkl(nn.Module):
    def __init__(self, input_dim, kernel, log_beta):
        super(cigp_dkl, self).__init__()
        self.kernel = kernel
        self.log_beta = nn.Parameter(torch.tensor([log_beta]))
        self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim *4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim *4, input_dim * 4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 4, input_dim * 4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 4, input_dim)).double()
        # self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim *4),
        #                                             nn.LeakyReLU(),
        #                                             nn.Linear(input_dim *4, input_dim * 4),
        #                                             nn.LeakyReLU(),
        #                                             nn.Linear(input_dim * 4, input_dim * 4),
        #                                             nn.LeakyReLU(),
        #                                             nn.Linear(input_dim * 4, input_dim * 4),
        #                                             nn.LeakyReLU(),
        #                                             nn.Linear(input_dim * 4, input_dim),
        #                                             ).double()


    def forward(self, data_manager, x_test, fidelity_indicator = None, normal = False):
        
        
        if fidelity_indicator is not None:
            x_test = torch.cat([x_test.reshape(-1,x_test.shape[1]),(torch.tensor(fidelity_indicator)+1).reshape(-1,1)], dim = 1)
        x_train, y_train = data_manager.get_data(0, normal=normal)
        x_train1 = self.FeatureExtractor(x_train)
        x_test1 = self.FeatureExtractor(x_test)

        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None
        Sigma = self.kernel(x_train1, x_train1) + self.log_beta.exp().pow(-1) * torch.eye(x_train1.size(0)).to(x_train1.device) \
            + JITTER * torch.eye(x_train1.size(0)).to(x_train1.device)
        
        kx = self.kernel(x_train1, x_test1)
        L = torch.linalg.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(y_train, L)  # torch.linalg.cholesky()
        
        var = self.kernel(x_test1, x_test1) - LinvKx.t() @ LinvKx

        # add the noise uncertainty
        var = var + self.log_beta.exp().pow(-1)
        # if y_train_var is not None:
        #     var = var + y_train_var.diag()* torch.eye(x_test.size(0))

        return mean, var

    def negative_log_likelihood(self, x_train, y_train):
        # x_train1 = self.FeatureExtractor(x_train)
        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None
        y_num, y_dimension = y_train.shape
        Sigma = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(
            x_train.size(0)).to(x_train.device) + JITTER * torch.eye(x_train.size(0)).to(x_train.device)
        if y_train_var is not None:
            Sigma = Sigma + y_train_var.diag()* torch.eye(x_train.size(0)).to(x_train.device)
        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        Gamma,_ = torch.triangular_solve(y_train, L, upper = False)
        #option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll
    
def train_GPdkl(GPmodel, data_manager, max_iter=100, lr_init=1e-1, normal = False):
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr = lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        xtr, ytr = data_manager.get_data(0, normal=normal)
        xtr = GPmodel.FeatureExtractor(xtr.double())
        loss = GPmodel.negative_log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()), end='\r')
    print('',end='\n')


if __name__ == "__main__":
    torch.manual_seed(1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # generate the data
    x = []
    y = []
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    xlow_indices = torch.sort(xlow_indices).values
    x_low = x_all[xlow_indices]
    x.append(torch.cat((x_low, torch.full((x_low.shape[0], 1), 1)), dim=1))

    xhigh1_indices = torch.randperm(500)[:300]
    xhigh1_indices = torch.sort(xhigh1_indices).values
    x_high1 = x_all[xhigh1_indices]
    x.append(torch.cat((x_high1, torch.full((x_high1.shape[0], 1), 2)), dim=1))

    xhigh2_indices = torch.randperm(500)[:250]
    xhigh2_indices = torch.sort(xhigh2_indices).values
    x_high2 = x_all[xhigh2_indices]
    x.append(torch.cat((x_high2, torch.full((x_high2.shape[0], 1), 3)), dim=1))

    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
    y_test = torch.sin(x_test)
    x_test = torch.cat((x_test, torch.full((x_test.shape[0], 1), 3)), dim=1)

    y_low = torch.sin(x_low) - 0.5 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
    y.append(y_low)
    y_high1 = torch.sin(x_high1) - 0.3 * torch.sin(2 * x_high1) + torch.rand(300, 1) * 0.1 - 0.05
    y.append(y_high1)
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 - 0.05
    y.append(y_high2)
    

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x.to(device), 'Y': y.to(device)}
    ]
    fidelity_num = len(initial_data)

    fidelity_manager = MultiFidelityDataManager(initial_data)
    myGP = cigp_dkl(input_dim=x.shape[1],kernel = kernel.SquaredExponentialKernel(), log_beta = 1.).to(device)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_GPdkl(myGP, fidelity_manager, max_iter=200, lr_init=1e-2)

    # debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        ypred, ypred_var = myGP(fidelity_manager,x_test)

    # debugger.logger.info('prepare to plot')
    x_test = x_test[:,0].reshape(-1,1)
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
    

