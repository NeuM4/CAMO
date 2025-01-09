import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import math
from Data_simulation.Cost_Function.cost_pow_10 import cost


class Branin():
    def __init__(self):
        self.x_dim = 2
        self.search_range = [[0, 1.5], [0, 1.5], [0, 1.5]]
        self.cost = cost(self.search_range[-1])
        self.b = 5.1 / (4 * math.pow(math.pi, 2))
        self.c = 5 / math.pi
        self.r = 6
        self.t = 1 / (8 * math.pi)

    def get_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = torch.cat((input_x.reshape(1, 2), input_s * torch.ones(1)[:, None]), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        else:
            x = torch.cat((input_x, input_s.reshape(input_x.shape[0], 1)), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        return -Y.reshape(-1, 1)

    def Initiate_data(self, num, seed):
        # tem = []
        torch.manual_seed(seed)
        # for i in range(self.x_dim):
        #     tt = torch.rand(num, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
        #     tem.append(tt)

        # xtr = torch.cat(tem, dim=1)
        # fidelity_indicator = torch.rand(num, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + \
        #                      self.search_range[-1][0]
        xtr_low = torch.rand(num[0], 2).double() * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
        xtr_mid = torch.rand(num[1], 2).double() * (self.search_range[1][1] - self.search_range[1][0]) + self.search_range[1][0]
        xtr_high = torch.rand(num[2], 2).double() * (self.search_range[2][1] - self.search_range[2][0]) + self.search_range[2][0]
        xtr = torch.cat((xtr_low, xtr_mid, xtr_high), dim=0)
        fidelity_indicator1 = torch.ones(num[0], 1) * 0.33
        fidelity_indicator2 = torch.ones(num[1], 1) * 0.67
        fidelity_indicator3 = torch.ones(num[2], 1) * 1
        fidelity_indicator = torch.cat((fidelity_indicator1, fidelity_indicator2, fidelity_indicator3), dim=0)

        total_num = num[0] + num[1] + num[2]
        ytr = self.get_data(xtr, fidelity_indicator).reshape(total_num, 1)

        return xtr.double(), ytr.double(), fidelity_indicator.double()
    
    def Initiate_data2(self, num, seed):
        ## for cfkg
        tem = []
        
        for i in range(self.x_dim):
            torch.manual_seed(seed+ i)
            tt = torch.rand(num, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        xtr = torch.cat(tem, dim=1)
        fidelity_indicator = torch.rand(num, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + \
                             self.search_range[-1][0]

        ytr = self.get_data(xtr, fidelity_indicator).reshape(num, 1)

        return xtr.double(), ytr.double(), fidelity_indicator.double()
    
    def find_max_value_in_range(self, seed):
        
        # Generate random points within the search range
        num_points = 1000
        
        tem = []
        
        for i in range(self.x_dim):
            torch.manual_seed(seed+ i +55)
            tt = torch.rand(num_points, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        x = torch.cat(tem, dim = 1)
        fidelity_indicator = torch.ones(num_points, 1)
        
        y = self.get_data(x, fidelity_indicator)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y, dim=0)

        return max_value.item(), x.reshape(-1,1), y.reshape(-1,1)


if __name__ == "__main__":
    data = Branin()
    xtr, ytr, fidelity_indicator = data.Initiate_data(8, 1)
    #cost is a tensor
    Cost = data.cost.compute_model_cost(dataset = ytr, s_index = fidelity_indicator)
