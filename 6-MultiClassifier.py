# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:36:15 2023

@author: Jfitz
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.utils.multiclass import plot_confusion_matrix
#get_ipython().run_line_magic('matplotlib', 'inline')

#current_path = os.getcwd()
file = 'featuresAll.csv'

data = pd.read_csv(file)

dataX = data.copy().drop(['class','userid'],axis=1)
dataY = data['class'].copy()

X_train, X_test, y_train, y_test = train_test_split(dataX,
                                    dataY, test_size=0.10,
                                    random_state=2019, stratify=dataY)

#Trainingset 10-fold cross validation
k_fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2018)

#========================= LOGISTIC REGRESSION ==================================
penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C,
            class_weight=class_weight, random_state=random_state,
                            solver=solver, n_jobs=n_jobs, multi_class='auto')


trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1,2])

model = logReg

for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
                                          ,y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:],         X_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index],         y_train.iloc[cv_index]

    model.fit(X_train_fold, y_train_fold)
    y_train_pred = model.predict(X_train_fold)
    trainingScores.append(accuracy_score(y_train_fold, y_train_pred))

    y_cv_pred = model.predict(X_cv_fold)
    predictionsBasedOnKFolds.loc[X_cv_fold.index,:] =         model.predict_proba(X_cv_fold)
    cvScores.append(accuracy_score(y_cv_fold, y_cv_pred))


preds = pd.concat([y_train,predictionsBasedOnKFolds], axis=1)
preds.columns = ['trueLabel','prediction_0','prediction_1','prediction_2']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()


print('Training Scores:', trainingScores)
print('CV Scores:', cvScores)
print('Mean Training Score:', np.mean(trainingScores))
print('Mean CV Score:', np.mean(cvScores))
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
preds['trueLabel'] = le.fit_transform(preds['trueLabel'])


print('Classification Report:')
#print(classification_report(preds['trueLabel'], preds.iloc[:,1:].idxmax(axis=1)))
print(classification_report(preds['trueLabel'], preds.iloc[:,1:].astype(float).idxmax(axis=1)))

#plot_confusion_matrix(model, X_test, y_test, normalize=None, cmap=plt.cm.Blues, values_format='.2f')
#plt.title('Confusion Matrix - Logistic Regression')
#plt.show()

# assuming you have already computed the confusion matrix
cm = confusion_matrix(model, X_test, y_test, normalize=None, cmap=plt.cm.Blues, values_format='.2f')

# plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')