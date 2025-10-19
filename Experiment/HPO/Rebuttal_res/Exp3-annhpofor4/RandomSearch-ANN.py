import argparse
import time
import os
import sys
import logging
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation,Input
from scikeras.wrappers import KerasClassifier
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split,cross_val_score

def ANN(optimizer='sgd', neurons=32, activation = 'relu', loss='categorical_crossentropy'):
    model = Sequential([
        Input(shape=(64,)),
        Dense(neurons, activation=activation),
        Dense(neurons, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def RS_ANN(seed):
    recording = {
                    "wallclocktime": [],
                    "err": [],
                }
    d = datasets.load_digits()
    X = d.data
    y = d.target
    n_iter_search= 100
    clf = KerasClassifier(model=ANN, verbose=0, random_state=seed)

    rf_params = {
        'model__optimizer': ['adam','sgd'],
        'model__neurons': sp_randint(10,100),
        'model__activation': ['relu', 'tanh'],
        'batch_size': sp_randint(16,64),
        'epochs': sp_randint(10,100),
    }

    Random = RandomizedSearchCV(clf, param_distributions=rf_params,
                                n_iter=n_iter_search, cv=2, scoring='accuracy',
                                verbose=2, random_state=0)
    Random.fit(X, pd.get_dummies(y).values)
    results = Random.cv_results_
    best_acc = 0
    t1 = time.time()
    for i in range(n_iter_search):
        acc = results['mean_test_score'][i]
        std = results['std_test_score'][i]
        fit_time = results['mean_fit_time'][i]
        params = results['params'][i]
        params['epochs'] = 100
        clf_evl = KerasClassifier(build_fn=ANN, **params, random_state=seed)
        acc = cross_val_score(clf_evl, X, pd.get_dummies(y).values, cv=2,scoring='accuracy')
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
    Exp_marker = "ANN-HPO"
    # Set up logging
    log_file_path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res', Exp_marker, 'RS_experiment.log')
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for seed in range(start, 10):
        logging.info(f'seed: {seed} start')
        record = RS_ANN(seed)
        df = pd.DataFrame(record)
        save_dir = '/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/ANN-HPO/'
        filename = f'RS-seed_{seed}_res.csv'
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
    pass