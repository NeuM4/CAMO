import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import time
import torch
import logging
from FidelityFusion_Models import *
from FidelityFusion_Models.DMF_CAR import *
from FidelityFusion_Models.DMF_CAR_dkl import *
from Data_simulation.Synthetic_MF_Function import *
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import *
from Acq_Rebuttal.Discrete import *
from sklearn.metrics import mean_squared_error, r2_score
from FidelityFusion_Models.MF_data import min_max_normalizer_2
import argparse
import random

MF_model_list = {'ResGP': ResGP, 'AR': AR, 'GP': cigp,
                 'DMF_CAR': DMF_CAR, 'DMF_CAR_dkl': DMF_CAR_dkl}
Acq_list = {'UCB': DMF_UCB, 'EI': DMF_EI, 'cfKG': DMF_KG}
Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester, 'Currin':Currin, 'Park':Park,'Branin': Branin}

def MF_BO_discrete(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    total_fidelity_num = exp_config['total_fidelity_num']
    initial_index = exp_config['initial_index']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']
    cost_type = exp_config['cost_type']
    # search_range = exp_config['search_range']

    '''prepare initial data'''
    data = data_model(cost_type,total_fidelity_num)
    index = initial_index
    xtr, ytr = data.get_discrete_data(index, seed)
    search_range = data.search_range[0]

    model_cost = data.cost
    data_max, data_test = data.find_max_value_in_range()
    recording = {"cost": [],
                    "SR": [],
                    "IR": [],
                    "rmse": [],
                    "r2":[],
                    "max_value":[],
                    "operation_time": []}
    
    x_normer_list = [min_max_normalizer_2(xtr[i], columns=[0, 1], min_value=0, max_value=total_fidelity_num - 1) for i in range(total_fidelity_num)]
    y_normer_list = [min_max_normalizer_2(ytr[i], columns=[0], min_value=0, max_value=total_fidelity_num - 1) for i in range(total_fidelity_num)]
    
    for i in range(total_fidelity_num):
        xtr[i] = x_normer_list[i].normalize(xtr[i])
        ytr[i] = y_normer_list[i].normalize(ytr[i])
        
    initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': xtr[0], 'Y': ytr[0]},
                    {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': xtr[1], 'Y': ytr[1]},
                    {'fidelity_indicator': 2, 'raw_fidelity_name': '2','X': xtr[2], 'Y': ytr[2]},
                    {'fidelity_indicator': 3, 'raw_fidelity_name': '3','X': xtr[3], 'Y': ytr[3]},
                    {'fidelity_indicator': 4, 'raw_fidelity_name': '4','X': xtr[4], 'Y': ytr[4]},
                    {'fidelity_indicator': 5, 'raw_fidelity_name': '5','X': xtr[5], 'Y': ytr[5]},
                    {'fidelity_indicator': 6, 'raw_fidelity_name': '6','X': xtr[6], 'Y': ytr[6]},
                    {'fidelity_indicator': 7, 'raw_fidelity_name': '7','X': xtr[7], 'Y': ytr[7]},
                    {'fidelity_indicator': 8, 'raw_fidelity_name': '8','X': xtr[8], 'Y': ytr[8]},
                    {'fidelity_indicator': 9, 'raw_fidelity_name': '9','X': xtr[9], 'Y': ytr[9]},
                    {'fidelity_indicator': 10, 'raw_fidelity_name': '10','X': xtr[10], 'Y': ytr[10]},
                    {'fidelity_indicator': 11, 'raw_fidelity_name': '11','X': xtr[11], 'Y': ytr[11]},
                    {'fidelity_indicator': 12, 'raw_fidelity_name': '12','X': xtr[12], 'Y': ytr[12]},
                    {'fidelity_indicator': 13, 'raw_fidelity_name': '13','X': xtr[13], 'Y': ytr[13]},
                    {'fidelity_indicator': 14, 'raw_fidelity_name': '14','X': xtr[14], 'Y': ytr[14]},
                ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fidelity_num)]

    flag = True
    i = 0
    while flag:
        logging.info(f'Starting iteration {i + 1}')
        T1 = time.time()
        if exp_config["MF_model"] == "DMF_CAR_dkl":
            GP_model = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num,input_dim = xtr[0].shape[1], kernel_list = kernel1, if_nonsubset = True)
        
        elif exp_config["MF_model"] in  ["DMF_CAR","AR","ResGP"]:
            GP_model = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num, kernel_list = kernel1, if_nonsubset = True)

        # Fit the Gaussian process model to the sampled points
        if exp_config["MF_model"] == "ResGP":
            train_ResGP(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        if exp_config["MF_model"] == "AR":
            train_AR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        # if exp_config["MF_model"] == "CAR":
        #     train_CAR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        # if exp_config["MF_model"] == "DMF_CAR":
        #     train_DMFCAR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        # if exp_config["MF_model"] == "DMF_CAR_dkl":
        #     train_DMFCAR_dkl(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        
        #cal the initial value
        if i == 0:
            x_fi = []
            for f in range(total_fidelity_num):
                fi_x = fidelity_manager.get_data(f)[0]
                fi_x = x_normer_list[f].denormalize(fi_x)
                x_fi.append(fi_x)
            x_train = torch.cat(x_fi, dim=0)
            if args.data_name in ["Branin","Park","Currin"]:
                y_high_for_train = data.get_data(x_train, torch.tensor([1]*x_train.shape[0]))
            else:
                y_high_for_train = data.get_data(x_train, 1)
            best_y_high_train = max(y_high_for_train)

            cost_list = [fidelity_manager.get_data(i)[1] for i in range(total_fidelity_num)]
            cost_iter = model_cost.compute_model_ConDis_cost(cost_list,total_fidelity_num)
            
            recording["cost"].append(cost_iter.item())
            recording["max_value"].append(max(y_high_for_train).item())
            recording["SR"].append((data_max - best_y_high_train).item())

            with torch.no_grad():
                test_da = data_test[:100]
                test_da = x_normer_list[-1].normalize(test_da.double())
                mu, var = GP_model(fidelity_manager, data_test[:100].double(), normal = False)
                mu = y_normer_list[-1].denormalize(mu)
            if args.data_name in ["Branin", "Park", "Currin"]:
                y_gt = data.get_data(data_test[:100], torch.tensor([1]*data_test[:100].shape[0]))
            else:
                y_gt = data.get_data(data_test[:100], total_fidelity_num - 1)
            rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1,1), mu.detach())))
            r2 = r2_score(y_gt.reshape(-1,1), mu.detach())
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
                                                                search_range = search_range,
                                                                xnorm = x_normer_list,
                                                                ynorm = y_normer_list,
                                                                seed=(seed + 1234 + i, i))
            
        elif exp_config["Acq_function"] == "EI":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                GP_model= GP_model,
                                                                cost = model_cost,
                                                                data_manager = fidelity_manager,
                                                                search_range = search_range,
                                                                model_name = exp_config["MF_model"],
                                                                xnorm = x_normer_list,
                                                                ynorm = y_normer_list,
                                                                seed= seed + i + 1234)
            
        elif exp_config["Acq_function"] == "cfKG":
            Acq_function = Acq_list[exp_config["Acq_function"]](GP_model=GP_model,
                                                                data_model=data,
                                                                cost=model_cost,
                                                                fidelity_num=total_fidelity_num,
                                                                data_manager=fidelity_manager,
                                                                xdim=xtr[0].shape[1],
                                                                search_range = search_range,
                                                                model_name = exp_config["MF_model"],
                                                                xnorm = x_normer_list,
                                                                ynorm = y_normer_list,
                                                                seed=seed + i + 1234)
        
        # Get the next point to evaluate
        new_x, new_s = Acq_function.compute_next()

        #Check if x is out of bounds
        for k in range(new_x.shape[1]):
            if new_x[0][k] < search_range[0]:
                new_x[0][k] = search_range[0] + random.uniform(0, 0.01)
            elif new_x[0][k] > search_range[1]:
                new_x[0][k] = search_range[1] - random.uniform(0, 0.01)

        new_y = data.get_data(new_x, new_s)

        logging.info(f"Optimization finished {i} times. New x: {new_x}, New s: {new_s}, New y: {new_y.item()}")
        new_x = x_normer_list[new_s].normalize(new_x.double())
        new_y = y_normer_list[new_s].normalize(new_y.double())
        fidelity_manager.add_data(raw_fidelity_name=str(new_s), fidelity_index=new_s, x=new_x, y=new_y)

        # Calculate evaluation indicators
        x_fi = []
        for f in range(total_fidelity_num):
            fi_x = fidelity_manager.get_data(f)[0]
            fi_x = x_normer_list[f].denormalize(fi_x)
            x_fi.append(fi_x)
        x_train = torch.cat(x_fi, dim=0)
         
        # x_train = torch.cat((fidelity_manager.get_data(0)[0], fidelity_manager.get_data(1)[0]), dim=0)
        
        if args.data_name in ["Branin","Park","Currin"]:
            y_high_for_train = data.get_data(x_train, torch.tensor([1]*x_train.shape[0]))
        else:
            y_high_for_train = data.get_data(x_train, 1)
        best_y_high_train = max(y_high_for_train)
        
        T2 = time.time()

        cost_list = [fidelity_manager.get_data(i)[1] for i in range(total_fidelity_num)]
        cost_iter = model_cost.compute_model_ConDis_cost(cost_list, total_fidelity_num)
        if cost_iter >= 150:
            flag = False
        recording["cost"].append(cost_iter.item())
        recording["max_value"].append(max(y_high_for_train).item())
        recording["SR"].append((data_max - best_y_high_train).item())

        with torch.no_grad():
            test_da = data_test[:100]
            test_da = x_normer_list[-1].normalize(test_da.double())
            mu, var = GP_model(fidelity_manager, data_test[:100].double(), normal = False)
            mu = y_normer_list[-1].denormalize(mu)
        if args.data_name in ["Branin", "Park", "Currin"]:
            y_gt = data.get_data(data_test[:100], torch.tensor([1]*data_test[:100].shape[0]))
        else:
            y_gt = data.get_data(data_test[:100], total_fidelity_num - 1)
        rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1,1), mu.detach())))
        r2 = r2_score(y_gt.reshape(-1,1), mu.detach())
        recording["IR"].append((data_max - max(mu)).item())
        recording['r2'].append(r2.item())
        recording['rmse'].append(rmse.item())
        recording["operation_time"].append(T2 - T1)
        i += 1

    return recording


if __name__ == '__main__':


    # data_name = "non_linear_sin"
    # data_name = "forrester"
    # data_name = "Branin"
    # data_name = "maolin1"
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="Park")
    parser.add_argument("--cost_type", type=str, default="pow_10")
    args = parser.parse_args()
    data_name = args.data_name
    
    Exp_marker = "Con_dis_15_fidelity"

    print("Data name: ", data_name)
    for mf_model in ["AR","ResGP"]:
    # for mf_model in ["DMF_CAR"]:
        # for acq in ["EI"]:
        for acq in ["UCB","EI","cfKG"]:
            for seed in [0,1,2,3,4,5,6,7,8,9]:
                try:
                    # Set up logging
                    log_file_path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results', Exp_marker, data_name, args.cost_type , f'{mf_model}_experiment_{acq}.log')
                    log_dir = os.path.dirname(log_file_path)

                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    
                    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                    # logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                    exp_config = {
                                'seed': seed,
                                'data_model': Data_list[args.data_name],
                                'cost_type':args.cost_type,
                                'MF_model': mf_model,
                                'Acq_function': acq,
                                'total_fidelity_num': 15,
                                'initial_index': {0: 2, 1: 2 , 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
                                                  10: 2, 11: 2, 12: 2, 13: 2, 14: 2},
                                # 'search_range': [0 , 1],   
                                'MF_iterations': 200,
                                'MF_learning_rate': 0.01,
                        }
                    logging.info(f'Config:{exp_config}')
                    logging.info(f'Starting experiment with data_name: {data_name}')
                    logging.info(f'Starting seed: {seed}')
                    logging.info(f'Using MF model: {mf_model}')
                    logging.info(f'Using acquisition function: {acq}')
                    record = MF_BO_discrete(exp_config)

                    path_csv = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'DMF', 'Exp_results',Exp_marker,
                                            data_name,exp_config['cost_type'])
                    if not os.path.exists(path_csv):
                        os.makedirs(path_csv)

                    df = pd.DataFrame(record)
                    df.to_csv(path_csv + '/'+ mf_model+'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                                index=False)

                ## 标准化的数据很奇怪 这个脚本暂时先不用

                except Exception as e:
                    logging.error(f"An error occurred with seed {seed}: {e}")