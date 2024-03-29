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

#file = '4HourlyFeatures.csv'
file = '24HrFeatures.csv'

#data = pd.read_csv(current_path + file)
#JFitz - Set to read file from same directory as code
df = pd.read_csv(file)
# Step 3: Prepare the data for modeling
#X = df[['f.mean', 'f.sd', 'f.propZeros']]
#y = df['class1']

X = df.copy().drop(['id','class','date','Category','counter','patientID'],axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Encode the categorical variable
#X_train = pd.get_dummies(X_train, columns=['class1'])
#X_test = pd.get_dummies(X_test, columns=['class1'])

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])

# =============================================================================
# # Fit the logistic regression model on the training data
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)
# 
# # Step 5: Evaluate the model's performance on the testing data
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# confusionLR = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# print(f" Logistic Regression Accuracy: {accuracy}")
# print(f"LR Confusion Matrix: \n{confusionLR}")
# print(f"LR Classification Report:\n{class_report}")
# 
# 
# # plot the confusion matrix as a heatmap
# #sns.heatmap(confusionLR, annot=True, cmap='Blues', fmt='g')
# #===========================================================================================
# 
# # split the data into training and testing sets
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # create a random forest classifier object
# rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# 
# # fit the model on the training data
# rfc.fit(X_train, y_train)
# 
# # predict on the test data
# y_pred = rfc.predict(X_test)
# 
# # evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Accuracy: {accuracy}")
# confusionRF = confusion_matrix(y_test, y_pred)
# print(f"RanDom Forest Confusion Matrix: \n{confusionRF}")
# class_report = classification_report(y_test, y_pred)
# print(f"Random Forest Classification Report:\n{class_report}")
# 
# #sns.heatmap(confusionRF, annot=True, cmap='Greens', fmt='g')
# =============================================================================
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

#sns.heatmap(conf_mat, annot=True, cmap='Reds', fmt='g')
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

#sns.heatmap(conf_mat, annot=True, cmap='Reds', fmt='g')
#==============================================================================================

# =============================================================================
# 
# #  Define the decision tree model
# dt_model = DecisionTreeClassifier()
# 
# dt_model.fit(X_train, y_train)
# 
# # Evaluate the model
# y_pred = dt_model.predict(X_test)
# 
# acc_score = accuracy_score(y_test, y_pred)
# conf_mat = confusion_matrix(y_test, y_pred)
# 
# print("Decision Tree Accuracy score:", acc_score)
# print("Decision Tree Confusion matrix:\n", conf_mat)
# class_report = classification_report(y_test, y_pred)
# print(f"Decision Tree Classification Report:\n{class_report}")
# plt.figure(figsize=(10,6))
# plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Depressive', 'Control', 'Schizophrenic'])
# plt.savefig('DecisionTreeClassifier.pdf', dpi=300)
# plt.show()
# 
# sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
# #===============================================================================================
# 
# 
# # Step 2: Define the neural network model
# model = keras.Sequential([
#     keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(3, activation='softmax')
# ])
# 
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 
# # Step 3: Train the model
# model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
# 
# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# 
# print(" Neural Network Test accuracy:", test_acc)
# 
# 
# 
# 
# =============================================================================










