# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:11:09 2023

@author: fitzgeraldj
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
#current_path = os.getcwd()
file = '24HrAgg.csv'

#data = pd.read_csv(current_path + file)
#JFitz - Set to read file from same directory as code
data = pd.read_csv(file)
#JFitz Drop redundant and string fields
dataX = data.copy().drop(['id','class1','date','Category','counter'],axis=1)
dataY = data['class1'].copy()

#----------------------------------------
#JFitz - The follopwing code is redundant:
#-----------------------------------------
#featuresToScale = dataX.drop(['userid'],axis=1).columns
#featuresToScale = dataX
#sX = pp.StandardScaler(copy=True)
#dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])
#scalingFactors = pd.DataFrame(data=[sX.mean_,sX.scale_],index=['Mean','StDev'],columns=featuresToScale)


X_train, X_test, y_train, y_test = train_test_split(dataX,
                                    dataY, test_size=0.20,
                                    random_state=2019, stratify=dataY)
# Generate some example data
#X_train = np.random.rand(1000, 10)
#y_train = np.random.randint(3, size=(1000, 1))

# Define the neural network architecture
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert the labels to one-hot encoded format
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=3)

# Train the model
model.fit(X_train, y_train_one_hot, epochs=30, batch_size=20)

# Generate some test data
#X_test = np.random.rand(100, 10)
#y_test = np.random.randint(3, size=(100, 1))

# Convert the test labels to one-hot encoded format
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=3)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


#It seems that the model's accuracy is still not very high. You can try the following:

#Increase the number of epochs to see if the accuracy
# improves. This can help the model to learn more from the data.

#Try different architectures for the model. You can add 
#more layers, increase or decrease the number of neurons in #
#each layer, and add regularization techniques such as dropout to prevent overfitting.

#Normalize the data before training the model. This can help 
#to improve the convergence rate of the model.

#Increase the number of samples in the dataset. This 
#can help the model to learn more from the data and improve its accuracy.

#Try a different model altogether. Sometimes, certain 
#models are better suited for certain types of data, and it is 
#possible that the current model is not the best fit for this particular dataset.
