import numpy as np
import pandas as pd
import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

from Data_simulation.Synthetic_MF_Function import *
import time
import os
import argparse


class SyntheticFunction:
    def __init__(self, data_config, seed):
        self.data_model = data_config["data_model"](cost_type = data_config["cost_type"], total_fidelity_num = data_config["total_fidelity_num"])
        self.model_cost = self.data_model.cost
        self.data_name = data_config["data_name"]
        if self.data_name == "Hartmann":
            self.x_range = [[0, 1] for i in range(6)]
            self.fidelity_range = (0, 1)
        elif self.data_name == "Branin":
            self.x_range = [[0, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif self.data_name == "Park":
            self.x_range = [[-1, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif self.data_name == "Currin":
            self.x_range = [[0, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif self.data_name == "non_linear_sin":
            self.x_range = [[0, 1.5]]
            self.fidelity_range = (0, 1)
        elif self.data_name == "forrester":
            self.x_range = [[0, 1.5]]
            self.fidelity_range = (0, 1)
        elif self.data_name == 'bohachevsky':
            self.x_range = [[-5, 5] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif self.data_name == 'borehole':
            self.x_range = [[0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110], [63.1, 116], [700, 820], 
                            [1120, 1680], [9855, 12045]]
            self.fidelity_range = (0, 1)
        elif self.data_name == 'colvile':
            self.x_range = [[-1, 1], [-1, 1], [-1,1], [-1,1]]
            self.fidelity_range = (0, 1)
        elif self.data_name == 'himmblau':
            self.x_range = [[-4, 4], [-4, 4]]
            self.fidelity_range = (0, 1)

        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        for i in range(len(self.x_range)):
            x = Float("x" + str(i + 1), self.x_range[i])
            cs.add([x])

        s = Float("s", self.fidelity_range)
        cs.add([s])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = [config["x" + str(i + 1)] for i in range(len(self.x_range))]
        s = config["s"]
        
        result = self.data_model.get_cmf_data(torch.tensor(x), torch.tensor(s))
        return -result
    
    def plot(self, x, s):
        return self.data_model.get_cmf_data(x, s)
    def find_max(self,):
        return self.data_model.find_max_value_in_range()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="non_linear_sin")
    max_dic = {'forrester': 50, 'non_linear_sin':0.033,'Branin': 55,'Currin': 14,'Park': 2.2,
               'himmblau':303.5,'bohachevsky': 72.15,'colvile':609.26,'borehole':244}
    Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester, 'Branin': Branin, 'Park':Park, "Currin":Currin,
                 'bohachevsky':bohachevsky,'borehole':borehole,'colvile':colvile,'himmblau':himmelblau}
    args = parser.parse_args()
    data_name = args.data_name
    max_theoretical = max_dic[data_name]
    # for seed in [1, 2, 3, 4]:
    for seed in range(30):
        recording = {"cost": [], "SR": [], 'operation_time':[],'time':[]}
        recording["SR"].append(max_theoretical)
        recording["cost"].append(0)
        t1 =time.time()

        data_config = {"data_name": data_name, "data_model": Data_list[data_name], "cost_type": "pow_10", "total_fidelity_num": 2}
        model = SyntheticFunction(data_config, seed = seed)

        # Scenario object specifying the optimization "environment"
        scenario = Scenario(model.configspace, deterministic=True, n_trials=180,seed = seed)

        # Now we use SMAC to find the best hyperparameters
        smac = HPOFacade(
            scenario,
            model.train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )

        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(model.configspace.get_default_configuration())
        print(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost: {incumbent_cost}")
        t2 = time.time()
        recording["time"].append(t2-t1)
        recording["operation_time"].append(t2-t1)


        # Store the results
    
        data = []
        data_y = []
        for k, v in smac.runhistory.items():
            t3 = time.time()
            config = smac.runhistory.get_config(k.config_id)
            x = [config["x" + str(i + 1)] for i in range(len(model.x_range))]
            s = [config["s"]]
            data.append(x + s)
            cost = model.model_cost.compute_gp_cost(torch.tensor(data))
            y = v.cost  # type: ignore # 因为y是相反数-y
            data_y.append(model.plot(torch.tensor(x), torch.tensor(model.fidelity_range[-1])).item())
            t4 = time.time()
            recording["SR"].append(max_theoretical - max(data_y))
            recording["cost"].append(cost.item())
            # recording["data"].append(x+s)
            # recording["y"].append(data_y[-1])
            recording["operation_time"].append(v.time)
            recording["time"].append(t4-t3)

        df = pd.DataFrame(recording)
        directory = 'smac_result/' + data_name + '/'+'improve/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(directory + '/smac_seed_' + str(seed) + '.csv', index=False)
