# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:19:36 2023

@author: fitzgeraldj
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
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import SVM
from sklearn.svm import SVC  
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.utils.multiclass import plot_confusion_matrix
#get_ipython().run_line_magic('matplotlib', 'inline')

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


penalty = 'l2'
C = 1.0
random_state = 2018
solver = 'newton-cg'

# create separate binary classifiers for each class
classifiers = {}
for class_label in [0, 1, 2]:
    logReg = LogisticRegression(penalty=penalty, C=C,
                                 random_state=random_state,
                                 solver=solver)

    # set the positive class to the current class label, and group the other classes as negative
    y_train_binary = (y_train == class_label).astype(int)
    
    # train the binary classifier for the current class
    logReg.fit(X_train, y_train_binary)
    
    # save the classifier for later use
    classifiers[class_label] = logReg

# Step 5: Evaluate the model's performance on the testing data
y_pred_prob = pd.DataFrame(data=[], index=X_test.index, columns=[0, 1, 2])
for class_label in [0, 1, 2]:
    # predict probabilities for the current class using the corresponding binary classifier
    y_pred_prob[class_label] = classifiers[class_label].predict_proba(X_test)[:, 1]

# get the predicted class label as the one with the highest probability
y_pred = y_pred_prob.idxmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f" Logistic Regression Accuracy: {accuracy}")
print(f"LR Confusion Matrix: \n{confusion_mat}")
print(f"LR Classification Report:\n{class_report}")

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])

preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])

preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsRandomForest = preds.copy()


# calculate precision-recall curve and average precision score for the positive class
positive_class_label = 1
precision, recall, thresholds = precision_recall_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])
average_precision = average_precision_score(y_test == positive_class_label, y_pred_prob[positive_class_label])
# Plot Precision Recall Curve
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('1. Logistic Regression PRC: Average Precision = {0:0.2f}'.format(
          average_precision))

# calculate ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test == positive_class_label, y_pred_prob[positive_class_label]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('1. Logistic Regression ROC:   AUC = {0:0.2f}'.format(roc_auc_score(y_test == positive_class_label, y_pred_prob[positive_class_label])))
plt.legend(loc="lower right")
plt.show()

#============================RANDOM Forest Classifier ==============================================
n_estimators = 10
max_features = 'auto'
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_leaf_nodes = None
bootstrap = True
oob_score = False
n_jobs = -1
random_state = 2018
class_weight = 'balanced'
# create separate binary classifiers for each class
classifiers = {}
for class_label in [0, 1, 2]:
    RFC = RandomForestClassifier(n_estimators=n_estimators,
        max_features=max_features, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
         n_jobs=n_jobs, random_state=random_state,
        )

    # set the positive class to the current class label, and group the other classes as negative
    y_train_binary = (y_train == class_label).astype(int)
    
    # train the binary classifier for the current class
    RFC.fit(X_train, y_train_binary)
    
    # save the classifier for later use
    classifiers[class_label] = RFC

# Step 5: Evaluate the model's performance on the testing data
y_pred_prob = pd.DataFrame(data=[], index=X_test.index, columns=[0, 1, 2])
for class_label in [0, 1, 2]:
    # predict probabilities for the current class using the corresponding binary classifier
    y_pred_prob[class_label] = classifiers[class_label].predict_proba(X_test)[:, 1]

# get the predicted class label as the one with the highest probability
y_pred = y_pred_prob.idxmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
confusionRF = confusion_matrix(y_test, y_pred)
print(f"RanDom Forest Confusion Matrix: \n{confusionRF}")
class_report = classification_report(y_test, y_pred)
print(f"Random Forest Classification Report:\n{class_report}")


trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])

preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsRandomForest = preds.copy()


# calculate precision-recall curve and average precision score for the positive class
positive_class_label = 1
precision, recall, thresholds = precision_recall_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])
average_precision = average_precision_score(y_test == positive_class_label, y_pred_prob[positive_class_label])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('2. Random Forest PRC: Average Precision = {0:0.2f}'.format(
          average_precision))

fpr, tpr, thresholds = roc_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])

areaUnderROC = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('2. Random Forest ROC:   AUC = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()


#============================XG Boost ==================================================


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

#===============================SVM Classifier ======================================================

# create separate binary classifiers for each class
classifiers = {}
for class_label in [0, 1, 2]:
    SVM = SVC(kernel='linear', C=1.0, random_state=random_state)

    # set the positive class to the current class label, and group the other classes as negative
    y_train_binary = (y_train == class_label).astype(int)
    
    # train the binary classifier for the current class
    SVM.fit(X_train, y_train_binary)
    
    # save the classifier for later use
    classifiers[class_label] = SVM

# Step 5: Evaluate the model's performance on the testing data
y_pred_prob = pd.DataFrame(data=[], index=X_test.index, columns=[0, 1, 2])
for class_label in [0, 1, 2]:
    # predict probabilities for the current class using the corresponding binary classifier
    y_pred_prob[class_label] = classifiers[class_label].predict_proba(X_test)[:, 1]

# get the predicted class label as the one with the highest probability
y_pred = y_pred_prob.idxmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy}")
confusionSVM = confusion_matrix(y_test, y_pred)
print(f"SVM Confusion Matrix: \n{confusionSVM}")
class_report = classification_report(y_test, y_pred)
print(f"SVM Classification Report:\n{class_report}")


trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])

preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()


# calculate precision-recall curve and average precision score for the positive class
positive_class_label = 1
precision, recall, thresholds = precision_recall_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])
average_precision = average_precision_score(y_test == positive_class_label, y_pred_prob[positive_class_label])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('2. SVM PRC: Average Precision = {0:0.2f}'.format(
          average_precision))

fpr, tpr, thresholds = roc_curve(y_test == positive_class_label, y_pred_prob[positive_class_label])

areaUnderROC = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('2. SVM ROC:   AUC = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()