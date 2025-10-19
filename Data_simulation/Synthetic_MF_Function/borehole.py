import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch

from Data_simulation.Cost_Function.cost_pow_10 import cost as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost as cost_linear
from Data_simulation.Cost_Function.cost_log import cost as cost_log
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}

_tau = 2*torch.pi

class borehole():
    def __init__(self, cost_type, total_fidelity_num):
        self.total_fidelity_num = total_fidelity_num
        self.x_dim = 8
        self.search_range = [[0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110], [63.1, 116], [700, 820], [1120, 1680], [9855, 12045],[0,1]]
        self.cost = cost_list[cost_type](self.search_range[-1])
        
    def w_h(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        w = torch.log10(9 * t + 1)
        return w
    def w_l(self, t):
        w = 1 - self.w_h(t)
        return w
    
    def _borehole_base(self, xx, a, b):
        rw, r, Tu, Hu, Tl, Hl, L, Kw = xx[:,0], xx[:,1], xx[:,2], xx[:,3], xx[:,4], xx[:,5], xx[:,6], xx[:,7]
        
        frac1 = a * Tu * (Hu - Hl)
        frac2a = 2*L*Tu / (torch.log(r/rw) * (rw**2) * Kw)
        frac2b = Tu / Tl
        frac2 = torch.log(r/rw) * (b + frac2a + frac2b)
        
        return frac1 / frac2
        
    def get_cmf_data(self, input_x, input_s):
        xtr = input_x
        y_l = self._borehole_base(xtr, 5, 1.5).reshape(-1,1)
        y_h = self._borehole_base(xtr, _tau, 1.0).reshape(-1,1)
        ytr_fid = self.w_l(input_s) * y_l + self.w_h(input_s) * y_h
        
        return ytr_fid
    
    def get_data(self, input_x, input_s):
        xtr = input_x
        y_l = self._borehole_base(xtr, 5, 1.5).reshape(-1,1)
        y_h = self._borehole_base(xtr, _tau, 1.0).reshape(-1,1)
        ytr_fid = self.w_l(input_s) * y_l + self.w_h(input_s) * y_h
        
        return ytr_fid
    
    def Initiate_data(self, index, seed):
        torch.manual_seed(seed)
        
        low = []
        for i in range(self.x_dim):
            tt = torch.rand(index[0], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            low.append(tt)
        xtr_low = torch.cat(low, dim=1)
        
        high = []
        for i in range(self.x_dim):
            tt = torch.rand(index[1], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            high.append(tt)
        xtr_high = torch.cat(high, dim=1)
        
        xtr = [xtr_low, xtr_high]

        ytr_low = self.get_cmf_data(xtr_low, 0)
        ytr_high = self.get_cmf_data(xtr_high, 1)
        ytr = [ytr_low, ytr_high]

        return xtr, ytr
    
    def find_max_value_in_range(self):
        
        torch.manual_seed(1)
        
        # Generate random points within the search range
        num_points = 1000
        
        tem = []
        
        for i in range(self.x_dim):
            torch.manual_seed(i +55)
            tt = torch.rand(num_points, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        x = torch.cat(tem, dim = 1)
        fidelity_indicator = torch.ones(num_points, 1)
        
        y = self.get_cmf_data(x, fidelity_indicator)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y, dim=0)

        return max_value.item(), x.reshape(-1, self.x_dim)
    
if __name__ == "__main__":
    data = borehole('pow_10',2)
    max_value, _ = data.find_max_value_in_range()
    ##max_value = 244
    xtr, ytr = data.Initiate_data({0: 10, 1: 4}, 1)
    pass