from sklearn import datasets
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
import argparse
import time
import os
import sys
import logging
import pandas as pd

def RF_HPO(seed):
    recording = {
                    "wallclocktime": [],
                    "err": [],
                }
    # t1 = time.time()
    d = datasets.load_digits()
    X = d.data
    y = d.target

    rf_params = {
        'n_estimators': sp_randint(10,100),
        "max_features":sp_randint(1,64),
        'max_depth': sp_randint(5,50),
        "min_samples_split":sp_randint(2,11),
        "min_samples_leaf":sp_randint(1,11),
        "criterion":['gini','entropy']
    }
    n_iter_search = 100 #number of iterations is set to 20, you can increase this number if time permits
    clf = RandomForestClassifier(random_state=seed)
    Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy',random_state=seed,verbose=2,return_train_score=True)
    Random.fit(X, y)
    # print(Random.best_params_)
    # print("Accuracy:"+ str(Random.best_score_))
    # print("\n每次迭代的准确率(平均交叉验证得分):")
    results = Random.cv_results_
    best_acc = 0
    t1 = time.time()
    for i in range(n_iter_search):
        acc = results['mean_test_score'][i]
        std = results['std_test_score'][i]
        fit_time = results['mean_fit_time'][i]
        params = results['params'][i]
        params['n_estimators']=100
        clf_evl = RandomForestClassifier(**params, random_state=seed)
        acc = cross_val_score(clf_evl, X, y, cv=3,scoring='accuracy')
        if(acc.mean() > best_acc):
            best_acc = acc.mean()
        recording["wallclocktime"].append((time.time()- t1 + fit_time*3).item())
        recording["err"].append((1-best_acc).item())
        logging.info(f"Iter {i+1:3d} | Acc = {acc.mean():.4f} ± {std:.4f} | Time = {fit_time:.2f}s | Params = {params}")
    return recording

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random search")
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()
    start = args.start_seed
    Exp_marker = "CAMO-HPO"
    # Set up logging
    log_file_path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res', Exp_marker, 'RF_experiment.log')
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for seed in range(start, 30):
        logging.info(f'seed: {seed} start')
        record = RF_HPO(seed)
        df = pd.DataFrame(record)
        save_dir = '/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/CAMO-HPO/'
        filename = f'RF-seed_{seed}_res.csv'
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
    pass