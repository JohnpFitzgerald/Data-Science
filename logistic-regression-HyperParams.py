# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:59:05 2023

@author: fitzgeraldj
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score

file = '24HrAgg.csv'

#data = pd.read_csv(current_path + file)
#JFitz - Set to read file from same directory as code
df = pd.read_csv(file)

X = df.copy().drop(['id','class1','date','Category','counter'],axis=1)
y = df['class1'].copy()

kf = KFold(n_splits=50, shuffle=True, random_state=42) # use k=5 for k-fold cross-validation


penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'newton-cg' #lbfgs' 
multi_class = 'multinomial'
max_iter=1000

logReg = LogisticRegression(penalty=penalty, C=C,
            class_weight=class_weight, random_state=random_state,
                            solver=solver, multi_class=multi_class, max_iter=max_iter)

# Define the models and parameters to use
models = {'logReg': logReg}

# Define the k-fold cross-validation and leave-one-out cross-validation parameters

loo = LeaveOneOut() # use leave-one-out cross-validation

# Loop through the models and perform cross-validation
for model_name, model in models.items():
    # Use k-fold cross-validation
    print(f'{model_name} using k-fold cross-validation:')
    accuracy_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(f'Average accuracy: {np.mean(accuracy_scores):.3f}')
    
    # Use leave-one-out cross-validation
    print(f'{model_name} using leave-one-out cross-validation:')
    accuracy_scores = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(f'Average accuracy: {np.mean(accuracy_scores):.3f}')
