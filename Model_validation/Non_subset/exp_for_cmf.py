import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models.CMF_CAR import ContinuousAutoRegression_large as CMF_CAR, train_CAR_large
from FidelityFusion_Models.CMF_CAR_dkl import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Model_validation.calculate_metrix import calculate_metrix
from Model_validation.Load_Mfdata import get_full_name_list_with_fidelity, generate_nonsubset_data

import torch
import time
import pandas as pd

all_data_name_list = ["colville", "nonlinearsin", "toal", "forrester",
                          "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
                          "p1", "p2", "p3", "p4", "p5",
                          "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
                          "maolin15",
                          "maolin19", "maolin20",
                          "shuo6", "shuo11", "shuo15", "shuo16",
                          "test3", "test4", "test5", "test6", "test7"]

data_name_list_dim1 = ["forrester","nonlinearsin","tl1","tl2","tl3","tl4",
                       "test3","test4","p1","p2","maolin1"]
data_name_list_dim2 = ["tl5","tl6","tl7","tl8","test5","p3","p4","maolin5","maolin6","maolin7","maolin8"
                       ,"maolin10","maolin12","maolin13","shuo6",]
data_name_list_dim2_01_02 = ["p5"]
data_name_list_dim3 = ["tl9","maolin15","shuo11"]
data_name_list_dim4 = ["colville",]
data_name_list_dim6 = ["test6","maolin19"]
data_name_list_dim8 = ["tl10","test7","maolin20","shuo15"]
data_name_list_dim10 = ["toal","shuo16"]
data_name_list_dim20 = ["test8"]
data_name_list_dim30 = ["test9"]

#test7 shuo6 范围不同
# p5的 -0.1到-0.2

#shuo11的seed1 GAR 矩阵分解nan
# data_name_list_dim3 = ["shuo11"]
# test_data_list = ["forrester",
#                           "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
#                           "p1", "p2", "p3", "p4", "p5",
#                           "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
#                           "maolin15",
#                           "maolin19", "maolin20",
#                           "shuo6", "shuo11", "shuo15", "shuo16",
#                           "test3", "test4", "test5", "test6", "test7"]
test_data_list = ["p3"]

false_test = ["forrester", "tl3","tl10","p1","p2"]

model_dic = {'CMF_CAR': CMF_CAR,'CMF_CAR_dkl': CMF_CAR_dkl}
train_dic = {'CMF_CAR': train_CAR_large,'CMF_CAR_dkl': train_CMFCAR_dkl}


if __name__ == '__main__':
        
    # method_list = ['DMF_CAR']
    method_list = ['CMF_CAR','CMF_CAR_dkl']
    all_data_name_with_fi_list = get_full_name_list_with_fidelity(data_name_list = test_data_list)   
    for _data_name in all_data_name_with_fi_list:
        print(_data_name)
        for method in method_list:
            print(method)
            for _seed in [0,1,2]:
                print(_seed)
                recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'time':[]}
                for _high_fidelity_num in [8, 16, 32, 64]:
                    torch.manual_seed(_seed)
                    
                    xtr, Ytr, xte, Yte = generate_nonsubset_data(_data_name, num_points = 450, n_train = 300, n_test = 300)
                    
                    x_low = xtr[0]
                    x_low = torch.cat((x_low, torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
                    y_low = Ytr[0]
                    x_high1 = xtr[1][:_high_fidelity_num]
                    x_high1 = torch.cat((x_high1, 2*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
                    y_high1 = Ytr[1][:_high_fidelity_num]
                    x_test = xte
                    x_test = torch.cat((x_test, 2*torch.ones(x_test.shape[0]).reshape(-1,1)), 1)
                    x = torch.cat((x_low, x_high1), 0)
                    y = torch.cat((y_low, y_high1), 0)
                    y_test = Yte
                
                    initial_data = [
                                    {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x.double(), 'Y': y.double()},
                                ]

                    T1 = time.time()
                    fidelity_manager = MultiFidelityDataManager(initial_data)
                    # kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
                    kernel_x = kernel.SquaredExponentialKernel()
                    
                    model = model_dic[method](input_dim=x.shape[1]-1, kernel_x=kernel_x, b_init=1.0)
                    
                    
                    max_iter = 200
                    lr = 1e-2
                    train_dic[method](model, fidelity_manager, max_iter=max_iter, lr_init=lr)

                    with torch.no_grad():
                        ypred, ypred_var = model(fidelity_manager,x_test.double())
                        
                    metrics = calculate_metrix(y_test = y_test, y_mean_pre = ypred.reshape(-1, 1), y_var_pre = ypred_var)

                    T2 = time.time()
                    recording['train_sample_num'].append(_high_fidelity_num)
                    recording['rmse'].append(metrics['rmse'])
                    recording['nrmse'].append(metrics['nrmse'])
                    recording['r2'].append(metrics['r2'])
                    # recording['nll'].append(metrics['nll'])
                    recording['time'].append(T2 - T1)

                # Note: Debug use
                # path_csv = os.path.join('Model_validation', 'Non_Subset', 'exp_results', str(_data_name))
                
                # Note: Powershell use
                path_csv = os.path.join('exp_results', str(_data_name)) 
                if not os.path.exists(path_csv):
                        os.makedirs(path_csv)

                record = pd.DataFrame(recording)
                record.to_csv(path_csv + '/' + method + '_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

                        