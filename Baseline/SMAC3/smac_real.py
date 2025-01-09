import numpy as np
import pandas as pd
import torch
import os
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

from Data_simulation.Real_Application.HeatedBlock import HeatedBlock
from Data_simulation.Real_Application.VibratePlate import VibPlate
import time
import argparse

class SyntheticFunction:
    def __init__(self, data_config, seed):
        self.data_model = data_config["data_model"](cost_type = data_config["cost_type"], total_fidelity_num = data_config["total_fidelity_num"])
        self.model_cost = self.data_model.cost
        self.data_name = data_config["data_name"]
        if self.data_name == "HeatedBlock":
            self.x_range = [[0.1, 0.4], [0.1, 0.4], [0, 2*np.pi]]
        elif self.data_name == "VibratePlate":
            self.x_range = [[100e9, 500e9], [0.2, 0.6], [6000, 10000]]

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
        return self.data_model.get_cmf_data(x, torch.tensor(s))
    def find_max(self,):
        return self.data_model.find_max_value_in_range()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="HeatedBlock")
    max_dic = {'HeatedBlock':2,'VibratePlate':250}
    Data_list = {'HeatedBlock':HeatedBlock,'VibratePlate':VibPlate}
    args = parser.parse_args()
    data_name = args.data_name
    max_theoretical = max_dic[data_name]
    # for seed in [1, 2, 3, 4]:
    for seed in range(30):
        recording = {"cost": [], "SR": [], 'operation_time':[],'time':[]}
        recording["SR"].append(max_theoretical)
        recording["cost"].append(0)
        t1 = time.time()

        data_config = {"data_name": data_name, "data_model": Data_list[data_name], "cost_type": "pow_10", "total_fidelity_num": 2}
        model = SyntheticFunction(data_config, seed = seed)

        # Scenario object specifying the optimization "environment"
        scenario = Scenario(model.configspace, deterministic=True, n_trials=100,seed=seed)

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

        recording["operation_time"].append(t2-t1)
        recording["time"].append(t2-t1)

        # Store the results
        data = []
        data_y = []
        for k, v in smac.runhistory.items():
            t3 = time.time()
            config = smac.runhistory.get_config(k.config_id)
            x = [config["x" + str(i + 1)] for i in range(len(model.x_range))]
            s = [config["s"]]
            data.append(x + s)
            cost = model.model_cost.compute_gp_cost(torch.tensor(data)).item()
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
        directory = 'smac_result/' + data_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv('smac_result/' + data_name + '/smac_seed_' + str(seed) + '.csv', index=False)
        # file_path = '/mnt/h/eda/nips2024/mfbo_v2/Rebuttal_Experiment/smac/smac_result/' + data_name + '/smac_seed_' + str(seed) + '.csv'
        # directory = os.path.dirname(file_path)

        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # df = pd.DataFrame(recording)
        # df.to_csv(file_path, index=False)
        # df = pd.DataFrame(recording)
        # df.to_csv('/mnt/h/eda/nips2024/mfbo_v2/Rebuttal_Experiment/smac/smac_result/' + data_name + '/smac_seed_' + str(seed) + '.csv', index=False)

