'''
CAMO-HPO 2025-7-26
'''
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import logging
import pandas as pd
import torch
from FidelityFusion_Models.CMF_CAR import *
from FidelityFusion_Models.MF_data import min_max_normalizer_2
from Acq_Rebuttal.Continue import *
from FidelityFusion_Models.GP_DMF import *
import GaussianProcess.kernel as kernel
import time
import random
from Data_simulation.Synthetic_MF_Function.RF_Classifier import RF_Classifier
from Data_simulation.Synthetic_MF_Function.ANN_Classifer import ANN_Classifier
MF_model_list = {'CMF_CAR':ContinuousAutoRegression_large,'GP':cigp}
Acq_list = {'UCB': CMF_UCB}
Data_list = {'RF_Classifier': RF_Classifier, 'ANN_Classifier': ANN_Classifier}
def HPO_exp1(exp_config):
    seed = exp_config['seed']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']
    torch.manual_seed(seed)
    recording = {
                    "wallclocktime": [],
                    "err": [],
                }
    data_model = exp_config["data_model"]
    time1 = time.time()
    data = data_model(total_fidelity_num = exp_config['fidelity_num'])
    xtr, ytr = data.initiate_data(exp_config['data_num'], seed)
    search_range = data.search_range
    model_cost = data.cost
    x = torch.stack(xtr, dim=0)
    y = torch.stack(ytr, dim=0)
    # x_normer = min_max_normalizer_2(x, columns=[i for i in range(x.shape[1]-1)],min_value=1, max_value=exp_config['fidelity_num'])
    # y_normer = min_max_normalizer_2(y, columns=[0], min_value=1, max_value=exp_config['fidelity_num'])
    # x = x_normer.normalize(x)
    # y = y_normer.normalize(y)
    initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x.double(), 'Y': y.double()}
                ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SquaredExponentialKernel()
    i = 0
    best_acc = 0
    while i < 100 - exp_config['data_num']:
        try:
            if exp_config["MF_model"] == "GP":
                GP_model = MF_model_list[exp_config["MF_model"]](kernel = kernel1, log_beta = 1.0)
                train_GP(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
            elif exp_config["MF_model"] == "CMF_CAR":
                GP_model = MF_model_list[exp_config["MF_model"]](kernel_x=kernel1)
                train_CAR_large(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
            if i == 0:
                x_train = fidelity_manager.get_data(0)[0]
                for k in range(exp_config['data_num']):
                    acc = data.get_acc(x_train[k], eval=True)
                    if acc > best_acc:
                        best_acc = acc
                recording['wallclocktime'].append(time.time() - time1)
                recording['err'].append((1-best_acc).item())
            Acq_function = Acq_list[exp_config["Acq_function"]](
                                                                    x_dimension=x.shape[1] - 1,
                                                                    posterior_function=GP_model,
                                                                    data_manager=fidelity_manager, 
                                                                    search_range=search_range,
                                                                    model_cost=model_cost,
                                                                    # norm = [x_normer, y_normer],
                                                                    norm = None,
                                                                    seed=[seed + i + 1234, i]
                                                                )
            new_x, new_s = Acq_function.compute_next()
            for k in range(new_x.shape[1]):
                if new_x[0][k] < search_range[k][0]:
                    new_x[0][k] = search_range[k][0]
                elif new_x[0][k] > search_range[k][1]:
                    new_x[0][k] = search_range[k][1]
            new_x = torch.cat((new_x, new_s.reshape(-1,1)), dim=1)
            new_x = torch.round(new_x)
            new_y = data.get_acc(new_x[0])
            
            # new_x = x_normer.normalize(new_x.double())
            # new_x = torch.cat((new_x, torch.tensor(new_s + 1).reshape(-1,1)), dim=1)
            
            # new_y = y_normer.normalize(new_y.double())
            fidelity_manager.add_data(raw_fidelity_name= '0', fidelity_index= 0, x=new_x, y=new_y.reshape(-1,1))

            # x_train = fidelity_manager.get_data(0)[0]
            # x_train = x_normer.denormalize(x_train)

            acc = data.get_acc(new_x[0], eval=True)
            if acc > best_acc:
                best_acc = acc
            logging.info(f"Optimization finished {i} times. New x: {new_x}, New y: {new_y}, Best acc: {best_acc.item()}")
            recording['wallclocktime'].append(time.time() - time1)
            recording['err'].append((1-best_acc).item())
            i += 1
        except Exception as e:
            logging.error(f"Error {e}")
            return recording
    return recording
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAMO-HPO")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--data_name", type=str, default="ANN_Classifier", choices=list(Data_list.keys()), help="Name of the data model to use")
    args = parser.parse_args()
    start = args.start_seed
    Exp_marker = "ANN-HPO"
    # Set up logging
    log_file_path = os.path.join(sys.path[-1], 'Exp_res', Exp_marker, 'GP_experiment.log')
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for seed in range(start, 10):
        exp_config = {
                            'seed': seed,
                            'fidelity_num': 5,
                            'data_model': Data_list[args.data_name],
                            'Acq_function': "UCB",
                            'MF_model': "GP",
                            'data_num': 10,
                            'MF_iterations': 200,
                            'MF_learning_rate': 0.01,
                            'args':args
                    }
        logging.info(f'Config:{exp_config}')
        record = HPO_exp1(exp_config)
        df = pd.DataFrame(record)
        save_dir = '/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/ANN-HPO/'
        filename = f'GP-seed_{seed}_res.csv'
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
    pass