import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats
#Random Forest
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation,Input
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping

def ANN(optimizer='sgd', neurons=32, activation = 'relu', loss='categorical_crossentropy'):
    model = Sequential([
        Input(shape=(64,)),
        Dense(neurons, activation=activation),
        Dense(neurons, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    d = datasets.load_digits()
    X = d.data
    y = d.target

    # clf = KerasClassifier(build_fn=ANN, optimizer='adam', neurons=32, batch_size=32, epochs=30, activation='relu', loss='categorical_crossentropy')
    # clf.fit(X,pd.get_dummies(y).values)
    # scores = cross_val_score(clf, X, pd.get_dummies(y).values, cv=3, scoring='accuracy')
    # print("Accuracy:"+ str(scores.mean()))

    n_iter_search=3
    clf = KerasClassifier(model=ANN, verbose=0)

    rf_params = {
        'model__optimizer': ['adam','sgd'],
        'model__neurons': sp_randint(10,100),
        'model__activation': ['relu', 'tanh'],
        'batch_size': sp_randint(16,64),
        'epochs': sp_randint(10,100),
    }

    Random = RandomizedSearchCV(clf, param_distributions=rf_params,
                                n_iter=n_iter_search, cv=3, scoring='accuracy',
                                verbose=2, random_state=0)
    Random.fit(X, pd.get_dummies(y).values)

    # clf = KerasClassifier(build_fn=ANN, verbose=0)
    # Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy',verbose=2,random_state=0,return_train_score=True)
    # Random.fit(X, pd.get_dummies(y).values)
    print(Random.best_params_)
    print("Accuracy:"+ str(Random.best_score_))
    for i in range(n_iter_search):
        mean_score = Random.cv_results_['mean_test_score'][i]
        std_score = Random.cv_results_['std_test_score'][i]
        fit_time = Random.cv_results_['mean_fit_time'][i]
        params = Random.cv_results_['params'][i]
        print(f"Iter {i+1}: acc = {mean_score:.4f} ± {std_score:.4f}, params = {params}")
    pass