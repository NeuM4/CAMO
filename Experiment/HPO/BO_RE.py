from sklearn import datasets
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
import argparse
import time
import os
import sys
import logging
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
import numpy as np
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

def Bayes_HPO(seed):
    recording = {
                    "wallclocktime": [],
                    "mse": [],
                }
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('./input/housing.csv', header=None, delimiter=r"\s+", names=column_names)
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

    rf_params = {
        'n_estimators': Integer(10,100),
        "max_features":Integer(1,64),
        'max_depth': Integer(5,50),
        "min_samples_split":Integer(2,11),
        "min_samples_leaf":Integer(1,11),
        "criterion":['squared_error','absolute_error']
    }
    n_iter_search = 100 #number of iterations is set to 20, you can increase this number if time permits
    clf = RandomForestRegressor(random_state=seed)
    Bayes = BayesSearchCV(clf,rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error',random_state=seed,verbose=2,return_train_score=True)
    Bayes.fit(x_scaled, y)
    results = Bayes.cv_results_
    best_mse = 1
    t1 = time.time()
    for i in range(n_iter_search):
        mse = -results['mean_test_score'][i]
        std = results['std_test_score'][i]
        fit_time = results['mean_fit_time'][i]
        params = results['params'][i]
        params['n_estimators']=100
        clf_evl = RandomForestRegressor(**params, random_state=seed)
        scores = cross_val_score(clf_evl, x_scaled, y, cv=3, scoring='neg_mean_squared_error')
        if(-scores.mean() < best_mse):
            best_mse = -scores.mean()
        recording["wallclocktime"].append((time.time()- t1 + fit_time*3).item())
        recording["mse"].append(best_mse)
        logging.info(f"Iter {i+1:3d} | MSE = {mse:.4f} ± {std:.4f} | Time = {fit_time:.2f}s | Params = {params}")
    return recording

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayes search")
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()
    start = args.start_seed
    Exp_marker = "RE-HPO"
    # Set up logging
    log_file_path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res', Exp_marker, 'Bayes_experiment.log')
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for seed in range(start, 20):
        logging.info(f'seed: {seed} start')
        record = Bayes_HPO(seed)
        df = pd.DataFrame(record)
        save_dir = '/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/RE-HPO/'
        filename = f'Bayes-seed_{seed}_res.csv'
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
    pass