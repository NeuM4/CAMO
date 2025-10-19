import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from Data_simulation.Cost_Function.cost_pow_10 import cost as cost_pow_10
from Data_simulation.Cost_Function.cost_log import cost as cost_log
from sklearn import datasets
cost_list = {'pow_10': cost_pow_10, 'log': cost_log}

class RF_Classifier():
    def __init__(self, total_fidelity_num):
        self.x_dim = 5
        self.search_range = [[5,50],[2,11],[1,11],[0,1],[1,64],[10,100]]
        self.cost = cost_list['log'](self.search_range[-1])
        self.fi_num = total_fidelity_num
        d = datasets.load_digits()
        self.X = d.data
        self.y = d.target

    def get_acc(self, config, eval=False):
        if eval:
            config = {
            'n_estimators': 100,
            'max_features': int(config[4]),
            'max_depth': int(config[0]),
            'min_samples_split': int(config[1]),
            'min_samples_leaf': int(config[2]),
            'criterion': 'gini' if int(config[3]) == 0 else 'entropy'
            }
        else:
            config = {
                'n_estimators': int(config[5]),
                'max_features': int(config[4]),
                'max_depth': int(config[0]),
                'min_samples_split': int(config[1]),
                'min_samples_leaf': int(config[2]),
                'criterion': 'gini' if int(config[3]) == 0 else 'entropy'
            }
        clf = RandomForestClassifier(n_estimators=config['n_estimators'], 
                                     max_features=config['max_features'], 
                                     max_depth=config['max_depth'], 
                                     min_samples_split=config['min_samples_split'], 
                                     min_samples_leaf=config['min_samples_leaf'], 
                                     criterion=config['criterion'],
                                     random_state=0)
        scores = cross_val_score(clf, self.X, self.y, cv=3, scoring='accuracy')
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

if __name__ == "__main__":
    rf_classifier = RF_Classifier(total_fidelity_num=5)
    xtr, ytr = rf_classifier.initiate_data(data_num=10, seed=42)
    for x, y in zip(xtr, ytr):
        print(f"Sample X: {x.numpy()}, Sample Y: {y.item()}")
    acc = rf_classifier.get_acc(xtr[0].numpy(), eval=True)
    print(f"Accuracy of the first sample: {acc:.4f}")
    pass
        
        
            