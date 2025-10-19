import torch
import math
import torch.nn as nn

class upper_confidence_bound(nn.Module):
    def __init__(self, x_dimension, fidelity_num, posterior_function, data_manager, search_range, seed, xnorm = None, ynorm = None):
        super(upper_confidence_bound, self).__init__()

        # 传入的用于计算的函数/参数
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.pre_func = posterior_function

        # select criteria
        self.beta = 0.2 * int(self.x_dimension) * math.log(int(seed[1]+1))
        self.gamma = 1.0
        self.data_manager = data_manager
        self.seed = seed[0]
        self.search_range = search_range

        # torch.manual_seed(self.seed)
        self.x_list = []
        for i in range(self.fidelity_num):
            torch.manual_seed(self.seed + i + 10086)
            tt = torch.rand(self.x_dimension) * (self.search_range[1] - self.search_range[0]) + self.search_range[0]
            self.x_list.append(nn.Parameter(tt.double()))
        self.x_list = torch.nn.ParameterList(self.x_list)
        
        self.x_norm = xnorm
        self.y_norm = ynorm

        # Optimizer para/target
        # self.x = nn.Parameter(torch.ones(x_dimension))

    def optimise_adam(self, fidelity_indicator, niteration=100, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_ucb(fidelity_indicator)
            loss.backward()
            optimizer.step()
            # self.x.data.clamp_(0.0, 1.0)
            print('iter', i, 'x:', self.x_list[fidelity_indicator], 'loss_negative_ucb:',loss.item(), end='\n')

    def negative_ucb(self, fidelity_indicator):
        
        # x = self.data_manager.normalizelayer[self.pre_func.fidelity_num-1].normalize_x(self.x)
        if self.x_norm != None:
            try:
                x = self.x_norm[fidelity_indicator].normalize(self.x_list[fidelity_indicator].reshape(-1,self.x_dimension))
            except:
                x = self.x_norm.normalize(self.x_list[fidelity_indicator].reshape(-1,self.x_dimension))
        else:
            x = self.x_list[fidelity_indicator].reshape(-1,self.x_dimension)
        mean, var = self.pre_func(self.data_manager, x, fidelity_indicator, normal=False)
        if self.y_norm != None:
            try:
                mean = self.y_norm[fidelity_indicator].denormalize(mean)
            except:
                mean = self.y_norm.denormalize(mean)
        # mean, var = self.data_manager.normalizelayer[self.pre_func.fidelity_num-1].denormalize(mean, var)
        ucb = mean + self.beta * var
        return -ucb

    def compute_next(self):
        N_UCB = []
        UCB_x = []
        
        for i in range(self.fidelity_num):
            print("x={}".format(self.x_list[i]))
            self.optimise_adam(fidelity_indicator = i, niteration=10, lr=0.01)
            UCB_x.append(self.x_list[i].detach())
            N_UCB.append(self.negative_ucb(fidelity_indicator=i))

        new_x = UCB_x[N_UCB.index(min(N_UCB))]

        new_s = self.fidelity_num - 1
            
        for i in range(self.fidelity_num):
            with torch.no_grad():
                if self.x_norm != None:
                    try:
                        norm_x = self.x_norm[i].normalize(new_x.reshape(-1,self.x_dimension))
                    except:
                        norm_x = self.x_norm.normalize(new_x.reshape(-1,self.x_dimension))
                else:
                    norm_x = new_x.reshape(-1,self.x_dimension)
                _, v = self.pre_func(self.data_manager, norm_x, i, normal=False)
            if self.beta * v > self.gamma:
                new_s = i
                break
            
        return new_x.reshape(1,-1), new_s


