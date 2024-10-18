import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import numpy as np

from botorch.test_functions.multi_fidelity import AugmentedHartmann
from Data_simulation.Cost_Function.cost_pow_10 import cost

class Hartmann():
    def __init__(self):
        self.x_dim = 6
        self.search_range = [[0, 1.5] for i in range(7)]

        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.problem = AugmentedHartmann(negate=True).to(**tkwargs)
        self.cost = cost(self.search_range[-1])

    def get_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = torch.cat((input_x.reshape(1, self.x_dim), input_s * torch.ones(1, 1)), dim=1)
            Y = self.problem(x)
            if len(Y.shape) == 1:
                d = Y.shape[0]
                Y = Y.reshape(1, d)
        else:
            if isinstance(input_x, np.ndarray):
                # 初始化数据
                input_x = torch.from_numpy(np.concatenate((input_x, input_s.reshape(input_x.shape[0], 1)), axis=1))
            else:
                input_x = torch.cat([input_x, input_s * torch.ones(1).reshape(1, 1)], dim=1)

            x = input_x
            Y = self.problem(x)
            # Y = torch.cat((y.reshape(-1,1),input_s * torch.ones(1).reshape(1, 1)),dim = 1)

        return Y

    def Initiate_data(self, num, seed):
        tem = []
        for i in range(self.x_dim):
            torch.manual_seed(seed + 217 + i)
            tt = torch.rand(num, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        xtr = torch.cat(tem, dim=1)
        # fidelity_indicator = torch.rand(num, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + self.search_range[-1][0]
        fidelity_indicator = torch.rand(num, 1) * 2
        
        ytr = self.get_data(xtr, fidelity_indicator).reshape(num, 1)

        return xtr.double(), ytr.double(), fidelity_indicator.double()


if __name__ == "__main__":
    data = Hartmann()
    x, y, f = data.Initiate_data(8, 1)
    print(y)