# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:39:07 2023

@author: Jfitz
"""
import pandas as pd
import numpy as np
from scipy import stats

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file = 'AllReadings.csv'
data = pd.read_csv(file)
# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])    
print(data.dtypes)
print(f"no of records:  {data.shape[0]}")
print(f"no of variables: {data.shape[1]}")



def newId(idVal):
    if idVal[:5] == 'condi':
        return 'Schizophrenic'
    elif idVal[:5] == 'patie':
        return 'Depressive'
    elif idVal[:5] == 'contr':
        return 'Control'
    else:
        return '*UNKNOWN*'
    
data['Category'] = data['id'].apply(newId)

print(data)    

if '*UNKNOWN*' in data['Category'].values:
    print("unknowns found")
    
# number of each category

#categoryCount = data['Category'].value_counts()

#print(categoryCount)    
#OUTPUT:
#[2106734 rows x 5 columns]
#Control          1019990
#Schizophrenic     551716
#Depressive        535028
#Name: Category, dtype: int64

#data['Hour'] = data['timestamp'].dt.hour    

#print(data)

#data.set_index('timestamp', inplace=True)

# =============================================================================

# =============================================================================

    # convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])

grouped = data.groupby(['Category','date'])
newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
newData = newData.reset_index()

    #print(newData.columns)

    #result = newData[['id',[('activity', 'mean'), ('activity', 'std'), ('activity', '<lambda_0>')]]]

   
    #result = newData[['id', 'activity']].groupby('id').agg(['mean', 'std', lambda x: np.percentile(x, 75)-np.percentile(x, 25)])
newData.columns = ['Category','date','f.mean', 'f.sd', 'f.propZeros']
    # Save the result to a CSV file
#newData['class1'] = newData['userid'].str[:5].apply(lambda x: 1 if x == 'condi' else 0) 
newData = newData[['Category','date','f.mean','f.sd','f.propZeros']]
newData = newData.loc[~((newData['f.mean'] == 0
                         ) & (newData['f.sd'] == 0))]
newData = newData.loc[~((newData['f.propZeros'] == 0))]    
print("***  All 3 groups Baseline input file created  ***")
newData['counter'] = newData.groupby('Category').cumcount() + 1
newData.to_csv('C:/mtu/project/All3-features.csv', index=False)

grouped = newData.groupby('Category')

for name, group in grouped:
    plt.plot(group['f.mean'], label=name)
    
plt.legend()
plt.xlabel('Days')
plt.ylabel('Average Activity')
plt.show()    


#subData = data[['Category','activity']]

#dailyMean = subData.groupby(['Category'])

#dailyMean.boxplot(figsize=(10,6))



#plt.show()

#grouped = newData.groupby('Category')
# assume your dataframe is called df
#grouped.plot(x='counter', y='f.mean', label='activity', figsize=(10,6))

# add legend, title and axis labels
#plt.legend()
#plt.title('Daily Activity by Category')
#plt.xlabel('Cumulative Days')
#plt.ylabel('Mean Activity')

# show the plot
#plt.show()

   
# number of each category

categoryCount = data['Category'].value_counts()

print(categoryCount)   
print(data) 
print(newData)

# calculate the z-scores for actvity data
#z_scores = np.abs(stats.zscore(data['activity']))
# define a theshold for outliers
#threshold = 1500

# calculate mean, median, mode, and standard deviation
mean = np.mean(data['activity'])
median = np.median(data['activity'])
std_dev = np.std(data['activity'])

# calculate range
range = np.max(data['activity']) - np.min(data['activity'])

# calculate percentiles
percentiles = np.percentile(data['activity'], [25, 50, 75])

# print the results
print('Mean:', mean)
print('Median:', median)
#print('Mode:', mode)
print('Standard deviation:', std_dev)
print('Range:', range)
print('25th, 50th, and 75th percentiles:', percentiles)



# remove rows with activity = 0
#data = data[data['activity'] != 0]

# calculate z-scores for activity data
z_scores = np.abs(stats.zscore(newData['f.mean']))

# define threshold for outliers
#lthreshold = 20
#hthreshold = 500
#data = data['f.mean' > 10]
# filter out rows with z-score > threshold
#newData = newData[z_scores <= hthreshold]
#newData = newData[z_scores >= lthreshold]
#newData = newData.loc[~((newData['f.mean'] < 20))] 
#newData = newData.loc[~((newData['f.mean'] >500))] 
# normalize activity data using min-max scaling
newData['zScore'] = (newData['f.mean'] - newData['f.mean'].min()) / (newData['f.mean'].max() - newData['f.mean'].min())

# print the normalized data
print(newData)
#newData = newData.loc[~((newData['f.propZeros'] == 0))]    
ControlData = newData.loc[((newData['Category']== 'Control'))]
plt.hist(ControlData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of CONTROL')
plt.ylabel('Frequency')
plt.title('Histogram of Control Activity daily averages')
plt.show()
DepressionData = newData.loc[((newData['Category']== 'Depressive'))]
plt.hist(DepressionData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of DEPRESSIVE')
plt.ylabel('Frequency')
plt.title('Histogram of Depression Activity daily averages')
plt.show()
SchizophreniaData = newData.loc[((newData['Category']== 'Schizophrenic'))]
plt.hist(SchizophreniaData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of SCHIZOPHRENIA')
plt.ylabel('Frequency')
plt.title('Histogram of Schizophrenic Activity daily averages')
plt.show()

# =============================================================================
# zscores = newData.groupby('Category')
# 
# for name, group in zscores:
#     plt.plot(group['zScore'], label=name)
#     
# plt.legend()
# plt.xlabel('z-Scores')
# plt.ylabel('lel')
# plt.show() 
# =============================================================================
