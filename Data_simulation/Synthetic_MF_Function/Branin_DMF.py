import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import math

from Data_simulation.Cost_Function.cost_pow_10 import cost as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost as cost_linear
from Data_simulation.Cost_Function.cost_log import cost as cost_log
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}

class Branin_DMF:
    def __init__(self, cost_type, total_fidelity_num):
        self.total_fidelity_num = total_fidelity_num
        self.x_dim = 2
        # self.search_range = [[-5, 10], [0, 15]] ##dataset_range
        self.search_range = [[0, 1.5], [0, 1.5], [0, 1.5]]
        self.cost = cost_list[cost_type](self.search_range[-1])


    def eval_fed_L0(self, xn):
        f2 = self.eval_fed_L1(1.2 * (xn + 2))
        f1 = f2 - 3 * xn[:,1] + 1
        return f1
    
    def eval_fed_L1(self, xn):
        f3 = self.eval_fed_L2(xn - 2)
        f2 = 10 * torch.sqrt(f3) + 2 * (xn[:,0] - 0.5) - 3 * (3 * xn[:,1] - 1) - 1
        return f2


    def eval_fed_L2(self, xn):
        x1 = xn[:,0]
        x2 = xn[:,1]
        term1 = -1.275 * torch.square(x1) / torch.square(torch.tensor(math.pi)) + 5 * x1 / torch.tensor(math.pi) + x2 - 6
        term2 = (10 - 5 / (4 * torch.tensor(math.pi))) * torch.cos(x1)
        f3 = torch.square(term1) + term2 + 10
        return f3


    def get_data(self, input_x, input_s):
        xtr = input_x
        if input_s == 0:
            ytr = self.eval_fed_L0(xtr)
        elif input_s == 1:
            ytr = self.eval_fed_L2(xtr)
        elif input_s == 2:
            ytr = self.eval_fed_L1(xtr)

        new_y = ytr

        if len(new_y.shape) == 1:
            d = new_y.shape[0]
            new_y = new_y.reshape(1, d)
        return new_y

    def Initiate_data(self, index, seed):

        torch.manual_seed(seed)
        xtr_low = torch.rand(index[0], 2).double() * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
        xtr_mid = torch.rand(index[1], 2).double() * (self.search_range[1][1] - self.search_range[1][0]) + self.search_range[1][0]
        xtr_high = torch.rand(index[2], 2).double() * (self.search_range[2][1] - self.search_range[2][0]) + self.search_range[2][0]
        xtr = [xtr_low, xtr_mid, xtr_high]

        ytr_low = self.get_data(xtr_low, 0).reshape(-1, 1)
        ytr_mid = self.get_data(xtr_mid, 1).reshape(-1, 1)
        ytr_high = self.get_data(xtr_high, 2).reshape(-1, 1)
        ytr = [ytr_low,ytr_mid,ytr_high]

        return xtr, ytr
    
    def find_max_value_in_range(self):
        
        # Generate random points within the search range
        num_points = 1000
        
        x_samples = torch.rand(num_points, self.x_dim) * (self.search_range[-1][1] - self.search_range[-1][0]) + self.search_range[-1][0]
        
        y_samples = self.get_data(x_samples, self.total_fidelity_num - 1)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y_samples[:, 0], dim=0)

        return max_value.item(),x_samples.reshape(-1,self.x_dim)
    
if __name__ == "__main__":
    data = Branin_DMF(3)
    xtr, ytr = data.Initiate_data({0: 10, 1: 4, 2: 2}, 1)
    Cost = data.cost.compute_model_cost(ytr)
    print(Cost)