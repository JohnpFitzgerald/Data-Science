# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:13:36 2023

@author: Jfitz
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import SGD

import xgboost as xgb
import lightgbm as lgb
import numpy as np

# Step 1: Load the data
file = '24HrFeatures.csv'
#file = '4HourlyFeatures.csv'
df = pd.read_csv(file)

#Split the data into features and target
groups = df['patientID'].copy() # patient IDs for LeaveOneGroupOut
#24 Hour data:
X = df.copy().drop(['id','class','date','Category','counter','patientID'], axis=1)
# Daily Segmented data:
#X = df.copy().drop(['id','class','date','Category','counter','segment','patientID'], axis=1)

#Target [0,1,2]
y = df['class'].copy()


#Prepare the data for modeling
scaler = StandardScaler()
X[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X[['f.mean', 'f.sd', 'f.propZeros']])

# # Create a LeaveOneGroupOut cross-validation object
logo = LeaveOneGroupOut()

#========================= LOGISTIC REGRESSION ==================================


# Define hyperparameters to search over
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear']
}

# Create logistic regression model
model = LogisticRegression(random_state=42)

# Use grid search to find optimal hyperparameters
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

# Print best hyperparameters
print(grid_search.best_params_)

# Train the model with optimal hyperparameters
#logistic_reg = LogisticRegression(random_state=42, **grid_search.best_params_)
#logistic_reg.fit(X, y)


#Train and evaluate the model using LeaveOneGroupOut cross-validation
acc_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    acc_scores.append(acc_score)
# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1))



#----------------------Decision Tree-------------------------------------

# Define Decision Tree hyperparameters
dt_params = {
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': {0:1, 1:1, 2:1}
}

# Define the Decision Tree classifier
dt_model = DecisionTreeClassifier(**dt_params)


#Train and evaluate the model using LeaveOneGroupOut cross-validation
acc_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    acc_scores.append(acc_score)

print(f"Decision Tree Accuracy scores: {acc_scores}")
print(f"Decision Tree Mean accuracy score: {sum(acc_scores)/len(acc_scores)}")

# plot a heatmap of the confusion matrix
y_pred = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train, y_train)
    y_pred.extend(dt_model.predict(X_test))
conf_mat = confusion_matrix(y, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.show()


#-----------------Random Forest Model-------------------------------------

# =============================================================================
# n_estimators = 10
# max_features = 'auto'
# max_depth = None
# min_samples_split = 2
# min_samples_leaf = 1
# min_weight_fraction_leaf = 0.0
# max_leaf_nodes = None
# bootstrap = True
# oob_score = False
# n_jobs = -1
# random_state = 2018
# class_weight = 'balanced'
# =============================================================================
rf_model = RandomForestClassifier(n_estimators=10, max_depth=None, criterion='gini', 
                                   min_samples_split=2, min_samples_leaf=1, 
                                   max_features='sqrt', bootstrap=True, 
                                   class_weight='balanced', random_state=2018)


# Train and evaluate the model using Leave-One-Patient-Out cross-validation
acc_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model and make predictions
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Compute accuracy and append to list
    acc_scores.append(accuracy_score(y_test, y_pred))
    
print(f"Random Forest Accuracy scores: {acc_scores}")
print(f"RF Mean accuracy score: {sum(acc_scores)/len(acc_scores)}")

# plot a heatmap of the confusion matrix
y_pred = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rf_model.fit(X_train, y_train)
    y_pred.extend(rf_model.predict(X_test))
conf_mat = confusion_matrix(y, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.show()

#--------------------------SVM Model ------------------------------------------


# Step 4: Define the SVM model and hyperparameters
svm_model = SVC(kernel='rbf', decision_function_shape='ovr')

# Define the hyperparameters to search
params = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
}

# Step 5: Create a GridSearchCV object and fit the data
grid_search = GridSearchCV(svm_model, param_grid=params, cv=5)
grid_search.fit(X, y)

# Step 6: Print the best parameters and the mean cross-validated score
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Train and evaluate the model using Leave-One-Patient-Out cross-validation
acc_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model and make predictions
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    # Compute accuracy and append to list
    acc_scores.append(accuracy_score(y_test, y_pred))
    
print(f"SVM Accuracy scores: {acc_scores}")
print(f"SVM Mean accuracy score: {sum(acc_scores)/len(acc_scores)}")

# plot a heatmap of the confusion matrix
y_pred = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    svm_model.fit(X_train, y_train)
    y_pred.extend(svm_model.predict(X_test))
conf_mat = confusion_matrix(y, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.show()

#---------------------Keras -------------------------------------------------
# Set the random seed for reproducibility
tf.random.set_seed(42)

# Define the hyperparameters
input_dim = X.shape[1] # the number of features in the input
learning_rate = 0.1
epochs = 100
batch_size = 128


# Define the model architecture
model = Sequential([
    Dense(64, input_dim=input_dim),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(32),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
opt = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train and evaluate the model using Leave-One-Patient-Out cross-validation
acc_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # One-hot encode the target variable
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values
    
    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    acc_scores.append(accuracy_score(np.argmax(y_test, axis=-1), y_pred))
    
print(f"Keras Neural Network Accuracy: {acc_scores}")
print(f"Keras Mean accuracy score: {sum(acc_scores)/len(acc_scores)}")


#-----------------------Light GBM -----------------------------------------
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'random_state': 42
}

lgb_model = lgb.LGBMClassifier(**params)

#objective: The objective function to be optimized.
#: The evaluation metric to be used.
#num_leaves: The maximum number of leaves in each tree.
#learning_rate: The learning rate.
#feature_fraction: The fraction of features to be used for each tree.
#bagging_fraction: The fraction of samples to be used for each tree.
#: The frequency of bagging.
#max_depth: The maximum depth of each tree.
#num_threads: The number of threads to use for training.

#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"LightGBM Accuracy scores: {accuracy_scores}")
print(f"Light GBM Mean accuracy score: {np.mean(accuracy_scores)}")

#--------------------------XG Boost ----------------------------------------
xgb_model = xgb.XGBClassifier(    
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8)

#n_estimators: The number of trees in the forest.
#learning_rate: The learning rate shrinks the contribution of each tree by learning_rate.
#max_depth: The maximum depth of each tree.
#subsample: The fraction of samples to be used for training each tree.
#: The fraction of features to be used for training each tree


#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"XGBoost Accuracy scores: {accuracy_scores}")
print(f"XGBoost Mean accuracy score: {np.mean(accuracy_scores)}")


