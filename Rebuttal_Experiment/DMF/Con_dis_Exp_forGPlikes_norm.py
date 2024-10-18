import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import time
import torch
import logging
from FidelityFusion_Models.CMF_CAR import *
from FidelityFusion_Models.CMF_CAR_dkl import *
from Data_simulation.Synthetic_MF_Function import *
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import *
from Acq_Rebuttal.Discrete import *
from FidelityFusion_Models.GP_DMF import *
from sklearn.metrics import mean_squared_error, r2_score
from FidelityFusion_Models.MF_data import min_max_normalizer_2
import argparse
import random


MF_model_list = {'CMF_CAR': ContinuousAutoRegression_large, 'CMF_CAR_dkl': CMF_CAR_dkl, "GP": cigp}
Acq_list = {'UCB': DMF_UCB, 'EI': DMF_EI, 'cfKG': DMF_KG}
Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester, 'Branin': Branin, 'Park': Park, "Currin": Currin}

def MF_BO_discrete(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    total_fidelity_num = exp_config['total_fidelity_num']
    initial_index = exp_config['initial_index']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']
    cost_type = exp_config['cost_type']
    args = exp_config['args']

    '''prepare initial data'''
    data = data_model(cost_type, total_fidelity_num=total_fidelity_num)
    index = initial_index
    xtr, ytr = data.get_discrete_data(index, seed)
    search_range = data.search_range[0]
    
    data_max, data_test = data.find_max_value_in_range()
    model_cost = data.cost
    recording = {"cost": [], "SR": [], "IR": [], "rmse": [], "r2": [], "max_value": [], "operation_time": []}
    
    x = []
    y = []
    for f in range(total_fidelity_num):
        x.append(torch.cat((xtr[f], torch.full((xtr[f].shape[0], 1), f + 1)), dim=1))
        y.append(ytr[f])
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    
    if args.data_name in ["Branin", "Park", "Currin"]:
        x_normer = min_max_normalizer_2(x, columns=[0, 1], min_value=0, max_value=total_fidelity_num)
    else:
        x_normer = min_max_normalizer_2(x, columns=[0], min_value=0, max_value=total_fidelity_num)
    y_normer = min_max_normalizer_2(y, columns=[0], min_value=0, max_value=total_fidelity_num)
    x = x_normer.normalize(x)
    y = y_normer.normalize(y)

    initial_data = [{'fidelity_indicator': 0, 'raw_fidelity_name': '0', 'X': x.double(), 'Y': y.double()}]
    
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SquaredExponentialKernel(length_scale=1., signal_variance=1.)

    flag = True
    i = 0
    while flag:
        logging.info(f'Starting iteration {i + 1}')
        T1 = time.time()
        
        # Train the model
        if exp_config["MF_model"] == "GP":
            GP_model = MF_model_list[exp_config["MF_model"]](kernel=kernel1, log_beta=1.0)
        elif exp_config["MF_model"] == "CMF_CAR":
            GP_model = MF_model_list[exp_config["MF_model"]](kernel_x=kernel1)
        elif exp_config["MF_model"] == "CMF_CAR_dkl":
            GP_model = MF_model_list[exp_config["MF_model"]](input_dim=x.shape[1] - 1, kernel_x=kernel1)
        
        if exp_config["MF_model"] == "CMF_CAR_dkl":
            train_CMFCAR_dkl(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
        elif exp_config["MF_model"] == "GP":
            train_GP(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
        else:
            train_CAR_large(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
                
        # cal the initial value
        if i == 0:
            x_train = fidelity_manager.get_data(0)[0]
            x_train = x_normer.denormalize(x_train)
            if args.data_name in ["Branin", "Park", "Currin"]:
                y_high_for_train = data.get_data(x_train[:, :-1], torch.tensor([1] * x_train.shape[0]))
            else:
                y_high_for_train = data.get_data(x_train[:, :-1], 1)
            best_y_high_train = max(y_high_for_train.reshape(-1, 1))
            cost_iter = model_cost.compute_model_ConDisGP_cost([fidelity_manager.get_data(0)[0]],total_fidelity_num)
            recording["cost"].append(cost_iter.item())
            recording["max_value"].append(max(y_high_for_train.reshape(-1, 1)).item())
            recording["SR"].append((data_max - best_y_high_train).item())

            with torch.no_grad():
                test_da = data_test[:100]
                test_da = x_normer.normalize(test_da.double())
                mu, var = GP_model(fidelity_manager, data_test[:100].double(), torch.tensor([1] * 100).double(), normal=False)
                mu = y_normer.denormalize(mu)
            
            if args.data_name in ["Branin", "Park", "Currin"]:
                y_gt = data.get_data(data_test[:100], torch.tensor([1] * data_test[:100].shape[0]))
            else:
                y_gt = data.get_data(data_test[:100], input_s=1)
            rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1, 1), mu.detach())))
            r2 = r2_score(y_gt.reshape(-1, 1), mu.detach())
            recording["IR"].append((data_max - max(mu)).item())
            recording['r2'].append(r2.item())
            recording['rmse'].append(rmse.item())
            recording["operation_time"].append(0)        
        
        
        # choose acquisition function
        if exp_config["Acq_function"] == "UCB":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                posterior_function=GP_model,
                                                                data_manager=fidelity_manager,
                                                                search_range=search_range,
                                                                seed=(seed + 1234 + i,i),
                                                                xnorm=x_normer,
                                                                ynorm=y_normer)
            
        elif exp_config["Acq_function"] == "EI":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                GP_model=GP_model,
                                                                cost=model_cost,
                                                                data_manager=fidelity_manager,
                                                                search_range=search_range,
                                                                model_name=exp_config["MF_model"],
                                                                seed=seed + i + 1234,
                                                                xnorm=x_normer,
                                                                ynorm=y_normer)
            
        elif exp_config["Acq_function"] == "cfKG":
            Acq_function = Acq_list[exp_config["Acq_function"]](GP_model=GP_model,
                                                                data_model=data,
                                                                cost=model_cost,
                                                                fidelity_num=total_fidelity_num,
                                                                data_manager=fidelity_manager,
                                                                xdim=xtr[0].shape[1],
                                                                search_range=search_range,
                                                                model_name=exp_config["MF_model"],
                                                                seed=seed + i + 1234,
                                                                xnorm=x_normer,
                                                                ynorm=y_normer)
        # Get the next point to evaluate
        new_x, new_s = Acq_function.compute_next()

        # Check if the new point is within the search range
        for k in range(new_x.shape[1]):
            if new_x[0][k] < search_range[0]:
                new_x[0][k] = search_range[0] + random.uniform(0, 0.01)
            elif new_x[0][k] > search_range[1]:
                new_x[0][k] = search_range[1] - random.uniform(0, 0.01)

        new_y = data.get_cmf_data(new_x, new_s)

        logging.info(f"Optimization finished {i} times. New x: {new_x}, New s: {new_s}, New y: {new_y.item()}")
        new_x = x_normer.normalize(new_x.double())
        new_x = torch.cat((new_x, torch.tensor(new_s + 1).reshape(-1, 1)), dim=1)
        new_y = y_normer.normalize(new_y.double())
        fidelity_manager.add_data(raw_fidelity_name='0', fidelity_index=0, x=new_x, y=new_y)

        # Calculate evaluation indicators
        x_train = fidelity_manager.get_data(0)[0]
        x_train = x_normer.denormalize(x_train)
        if args.data_name in ["Branin", "Park", "Currin"]:
            y_high_for_train = data.get_data(x_train[:, :-1], torch.tensor([1] * x_train.shape[0]))
        else:
            y_high_for_train = data.get_data(x_train[:, :-1], 1)
        best_y_high_train = max(y_high_for_train.reshape(-1, 1))
        
        T2 = time.time()

        cost_iter = model_cost.compute_model_ConDisGP_cost([fidelity_manager.get_data(0)[0]],total_fidelity_num)
        if cost_iter >= 150:
            flag = False
        recording["cost"].append(cost_iter.item())
        recording["max_value"].append(max(y_high_for_train.reshape(-1, 1)).item())
        recording["SR"].append((data_max - best_y_high_train).item())

        with torch.no_grad():
            test_da = data_test[:100]
            test_da = x_normer.normalize(test_da.double())
            mu, var = GP_model(fidelity_manager, data_test[:100].double(), torch.tensor([1] * 100).double(), normal=False)
            mu = y_normer.denormalize(mu)
            
        if args.data_name in ["Branin", "Park", "Currin"]:
            y_gt = data.get_data(data_test[:100], torch.tensor([1] * data_test[:100].shape[0]))
        else:
            y_gt = data.get_data(data_test[:100], input_s=1)
        rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1, 1), mu.detach())))
        r2 = r2_score(y_gt.reshape(-1, 1), mu.detach())
        recording["IR"].append((data_max - max(mu)).item())
        recording['r2'].append(r2.item())
        recording['rmse'].append(rmse.item())
        recording["operation_time"].append(T2 - T1)
        i += 1

    return recording

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="forrester")
    parser.add_argument("--cost_type", type=str, default="pow_10")
    args = parser.parse_args()
    data_name = args.data_name
    Exp_marker = "Norm_res"
    
    # Set up logging
    log_file_path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results', Exp_marker, data_name, args.cost_type , f'experiment_GP.log')
    log_dir = os.path.dirname(log_file_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    for seed in range(10):
    # seed = 4
        for mf_model in ['GP']:
            for acq in ["UCB", "EI", "cfKG"]:
                try:
                    exp_config = {
                        'seed': seed,
                        'data_model': Data_list[args.data_name],
                        'cost_type': args.cost_type,
                        'MF_model': mf_model,
                        'Acq_function': acq,
                        'total_fidelity_num': 10,
                        'initial_index': {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3},
                        'MF_iterations': 200,
                        'MF_learning_rate': 0.01,
                        'args': args
                    }
                    logging.info(f'Config:{exp_config}')
                    logging.info(f'Starting experiment with data_name: {data_name}')
                    logging.info(f'Starting seed: {seed}')
                    logging.info(f'Using MF model: {mf_model}')
                    logging.info(f'Using acquisition function: {acq}')
                    
                    record = MF_BO_discrete(exp_config)
                    path_csv = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results', Exp_marker, data_name, exp_config['cost_type'])
                    if not os.path.exists(path_csv):
                        os.makedirs(path_csv)
                    df = pd.DataFrame(record)
                    df.to_csv(path_csv + '/' + mf_model + '_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv', index=False)
                
                except Exception as e:
                    logging.error(f"An error occurred with seed {seed}: {e}")
