import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from Data_simulation.Cost_Function.cost_pow_10 import cost as cost_pow_10
from Data_simulation.Cost_Function.cost_log import cost as cost_log
from sklearn import datasets
from pandas import read_csv
from sklearn import preprocessing
import pandas as pd
import numpy as np
cost_list = {'pow_10': cost_pow_10, 'log': cost_log}

class RF_Regressor():
    def __init__(self, total_fidelity_num):
        self.x_dim = 5
        self.search_range = [[5,50],[2,11],[1,11],[0,1],[1,13],[10,100]]
        self.cost = cost_list['log'](self.search_range[-1])
        self.fi_num = total_fidelity_num
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = read_csv('/home/fillip/桌面/CAMO/Data_simulation/Real_Application/housing.csv', header=None, delimiter=r"\s+", names=column_names)
        min_max_scaler = preprocessing.MinMaxScaler()
        column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
        x = data.loc[:,column_sels]
        y = data['MEDV']
        x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
        y =  np.log1p(y)
        for col in x.columns:
            if np.abs(x[col].skew()) > 0.3:
                x[col] = np.log1p(x[col])
        
        x_scaled = min_max_scaler.fit_transform(x)
        self.X = x_scaled
        self.y = y
    
    def get_mse(self, config, eval=False):
        if eval:
            config = {
                'n_estimators': 100,
                'max_features': int(config[4]),
                'max_depth': int(config[0]),
                'min_samples_split': int(config[1]),
                'min_samples_leaf': int(config[2]),
                'criterion': 'squared_error' if int(config[3]) == 0 else 'absolute_error'
            }
        else:
            config = {
                'n_estimators': int(config[5]),
                'max_features': int(config[4]),
                'max_depth': int(config[0]),
                'min_samples_split': int(config[1]),
                'min_samples_leaf': int(config[2]),
                'criterion': 'squared_error' if int(config[3]) == 0 else 'absolute_error'
            }
        clf = RandomForestRegressor(n_estimators=config['n_estimators'], 
                                     max_features=config['max_features'], 
                                     max_depth=config['max_depth'], 
                                     min_samples_split=config['min_samples_split'], 
                                     min_samples_leaf=config['min_samples_leaf'], 
                                     criterion=config['criterion'],
                                     random_state=0)
        scores = cross_val_score(clf, self.X, self.y, cv=3, scoring='neg_mean_squared_error')
        return torch.tensor([-scores.mean()])
    
    def initiate_data(self, data_num, seed):
        torch.manual_seed(seed)
        xtr = []
        ytr = []
        for i in range(data_num):
            sample_x = []
            for i,(low, high) in enumerate(self.search_range):
                value = torch.randint(low, high+1,size=(1,)).item()
                sample_x.append(value)
            mse = self.get_mse(sample_x)
            sample_x = torch.tensor(sample_x, dtype=torch.float32)
            xtr.append(sample_x)
            ytr.append(torch.tensor([mse], dtype=torch.float32))
        return xtr, ytr

if __name__ == "__main__":
    rf_regressor = RF_Regressor(total_fidelity_num=5)
    xtr, ytr = rf_regressor.initiate_data(data_num=10, seed=42)
    print("Sample training data:", xtr[:5])
    print("Sample target values:", ytr[:5])
    mse = rf_regressor.get_mse(xtr[0].numpy(), eval=True)
    print(f"MSE of the first sample: {mse.item():.4f}")
    pass