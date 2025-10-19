import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from Data_simulation.Cost_Function.cost_pow_10 import cost as cost_pow_10
from Data_simulation.Cost_Function.cost_log import cost as cost_log
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
import pandas as pd
cost_list = {'pow_10': cost_pow_10, 'log': cost_log}

def ANN(optimizer='sgd', neurons=32, activation = 'relu', loss='categorical_crossentropy'):
    model = Sequential([
        Input(shape=(64,)),
        Dense(neurons, activation=activation),
        Dense(neurons, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

class ANN_Classifier():
    def __init__(self, total_fidelity_num):
        self.x_dim = 4
        self.search_range = [[0,1],[10,100],[0,1],[16,64],[10,100]]
        self.cost = cost_list['log'](self.search_range[-1])
        self.fi_num = total_fidelity_num
        d = datasets.load_digits()
        self.X = d.data
        self.y = d.target

    def get_acc(self, config, eval=False):
        if eval:
            config = {
                'model__optimizer': 'adam' if int(config[0]) == 0 else 'sgd',
                'model__neurons': int(config[1]),
                'batch_size': int(config[3]),
                'epochs': 100,
                'model__activation': 'relu' if int(config[2]) == 0 else 'tanh',
            }
        else:
            config = {
                'model__optimizer': 'adam' if int(config[0]) == 0 else 'sgd',
                'model__neurons': int(config[1]),
                'batch_size': int(config[3]),
                'epochs': int(config[4]),
                'model__activation': 'relu' if int(config[2]) == 0 else 'tanh',
            }
        
        clf = KerasClassifier(build_fn=ANN, **config, random_state=0)
        scores = cross_val_score(clf, self.X, pd.get_dummies(self.y).values, cv=2, scoring='accuracy')
        return torch.tensor([scores.mean()])
    
    def initiate_data(self, data_num, seed):
        torch.manual_seed(seed)
        xtr = []
        ytr = []
        for i in range(data_num):
            sample_x = []
            for i,(low, high) in enumerate(self.search_range):
                value = torch.randint(low, high+1,size=(1,)).item()
                sample_x.append(value)
            acc = self.get_acc(sample_x)
            sample_x = torch.tensor(sample_x, dtype=torch.float32)
            xtr.append(sample_x)
            ytr.append(torch.tensor([acc], dtype=torch.float32))
        return xtr, ytr