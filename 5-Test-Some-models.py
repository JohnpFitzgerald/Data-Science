# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:12:29 2023
@author: Jfitz
"""
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

file = '24HrAgg.csv'

#data = pd.read_csv(current_path + file)
#JFitz - Set to read file from same directory as code
df = pd.read_csv(file)
# Step 3: Prepare the data for modeling
#X = df[['f.mean', 'f.sd', 'f.propZeros']]
#y = df['class1']

X = df.copy().drop(['id','class1','date','Category','counter'],axis=1)
y = df['class1'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Encode the categorical variable
#X_train = pd.get_dummies(X_train, columns=['class1'])
#X_test = pd.get_dummies(X_test, columns=['class1'])

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])



# =============================================================================
# penalty = 'l2'
# C = 1.0
# random_state = 2018
# solver = 'newton-cg'
# 
# logReg = LogisticRegression(penalty=penalty, C=C,
#              random_state=random_state,
#                             solver=solver)
# 
# model = logReg
# 
# trainingScores = []
# cvScores = []
# predictionsBasedOnKFolds = pd.DataFrame(data=[],
#                                         index=y_train.index,columns=[0,1,2])
# 
# #Trainingset 10-fold cross validation
# k_fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2018)
# 
# # Step 4: Fit the logistic regression model on the training data
# 
# model.fit(X_train, y_train)
# 
# # Step 5: Evaluate the model's performance on the testing data
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# confusion_mat = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# print(f" Logistic Regression Accuracy: {accuracy}")
# print(f"LR Confusion Matrix: \n{confusion_mat}")
# print(f"LR Classification Report:\n{class_report}")
# =============================================================================

# =============================================================================
# for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
#                                           ,y_train.ravel()):
#     X_train_fold, X_cv_fold = X_train.iloc[train_index,:],         X_train.iloc[cv_index,:]
#     y_train_fold, y_cv_fold = y_train.iloc[train_index],         y_train.iloc[cv_index]
# 
#     model.fit(X_train_fold, y_train_fold)
#     loglossTraining = log_loss(y_train_fold,
#                                model.predict_proba(X_train_fold)[:,1])
#     trainingScores.append(loglossTraining)
# 
#     predictionsBasedOnKFolds.loc[X_cv_fold.index,:] =         model.predict_proba(X_cv_fold)
#     loglossCV = log_loss(y_cv_fold,
#                          predictionsBasedOnKFolds.loc[X_cv_fold.index,1])
#     cvScores.append(loglossCV)
# =============================================================================


# =============================================================================
# for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
#     X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
#     y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
# 
#     model.fit(X_train_fold, y_train_fold)
#     loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold), labels=np.unique(y_train))
# 
#     trainingScores.append(loglossTraining)
# 
#     predictionsBasedOnKFolds.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
#     loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index], labels=np.unique(y_train))
#     cvScores.append(loglossCV)
# 
# preds = pd.concat([y_train,predictionsBasedOnKFolds], axis=1)
# preds.columns = ['trueLabel','prediction0', 'prediction1', 'prediction2']
# predictionsBasedOnKFoldsLogisticRegression = preds.copy()
# 
# 
# precision, recall, thresholds = precision_recall_curve(preds['trueLabel'], preds['prediction1'])
# average_precision = average_precision_score(preds['trueLabel'], preds['prediction1'])
# =============================================================================

#===========================================================================================

# split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a random forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the model on the training data
rfc.fit(X_train, y_train)

# predict on the test data
y_pred = rfc.predict(X_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
confusionRF = confusion_matrix(y_test, y_pred)
print(f"RanDom Forest Confusion Matrix: \n{confusionRF}")
class_report = classification_report(y_test, y_pred)
print(f"Random Forest Classification Report:\n{class_report}")


#================================================================================================


# Define the model
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("XGB Accuracy score:", acc_score)
print("XGB Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"XGB Classification Report:\n{class_report}")
#feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
#plt.barh(feat_imp.index, feat_imp.values)
#plt.title("XGB Feature importances")
#plt.show()


#================================================================================================


# Define the model
lgb_model = lgb.LGBMClassifier()

lgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lgb_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Light GBM Accuracy score:", acc_score)
print("Light GBM Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"Light GBM Classification Report:\n{class_report}")
#feat_imp = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
#plt.barh(feat_imp.index, feat_imp.values)
#plt.title("Light GBM Feature importances")
#plt.show()


#==============================================================================================


#  Define the decision tree model
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Decision Tree Accuracy score:", acc_score)
print("Decision Tree Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"Decision Tree Classification Report:\n{class_report}")
#plt.figure(figsize=(10,6))
#plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Depressive', 'Control', 'Schizophrenic'])
#plt.savefig('DecisionTreeClassifier.pdf', dpi=300)
#plt.show()


#===============================================================================================


# Step 2: Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 3: Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(" Neural Network Test accuracy:", test_acc)

