# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:39:07 2023

@author: Jfitz
"""
import pandas as pd
import numpy as np
from scipy import stats
import math
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
z_scores = np.abs(stats.zscore(data['activity']))

# define threshold for outliers
lthreshold = 100
hthreshold = 1000
#data = data['f.mean' > 10]
# filter out rows with z-score > threshold
#data = data[z_scores <= hthreshold]
#data = data[z_scores >= lthreshold]
data = data.loc[~((data['activity'] < 100))] 
data = data.loc[~((data['activity'] >1500))] 
# normalize activity data using min-max scaling
data['zScore'] = (data['activity'] - data['activity'].min()) / (data['activity'].max() - data['activity'].min())

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
ControlData = data.loc[((data['Category']== 'Control'))]
plt.hist(ControlData['activity'], bins=100)
plt.xlabel('Daily Activity of CONTROL')
plt.ylabel('Frequency')
plt.title('Histogram of Control Activity')
plt.show()
DepressionData = data.loc[((data['Category']== 'Depressive'))]
plt.hist(DepressionData['activity'], bins=100)
plt.xlabel('Dail Activity of DEPRESSIVE')
plt.ylabel('Frequency')
plt.title('Histogram of Depression Activity')
plt.show()
SchizophreniaData = data.loc[((data['Category']== 'Schizophrenic'))]
plt.hist(SchizophreniaData['activity'], bins=100)
plt.xlabel('Daily Activity of SCHIZOPHRENIA')
plt.ylabel('Frequency')
plt.title('Histogram of Schizophrenic Activity')
plt.show()




# =============================================================================
# 
def entropy(d):
     # count the frequency of each unique value in the data set
     freq_dict = {}
     for val in d:
         freq_dict[val] = freq_dict.get(val, 0) + 1
     
     # compute the probability of each unique value
     prob_dict = {}
     for val, freq in freq_dict.items():
         prob_dict[val] = freq / len(d)
     
     # compute the entropy of the data set
     entropy = 0
     for prob in prob_dict.values():
         entropy -= prob * math.log2(prob)
     
     return entropy
 
print(entropy(data['activity']))
# =============================================================================



# =============================================================================
# =============================================================================
# def fractal_dimension(d):
#      # generate a range of box sizes
#      box_sizes = np.floor(np.logspace(0, np.log2(len(data)), num=20))
#      
#      # count the number of boxes that contain at least one data point for each box size
#      box_counts = []
#      for size in box_sizes:
#          if size > 0:
#             count = 0
#             for i in range(0, len(data), size):
#                if np.sum(data[i:i+size]) > 0:
#                   count += 1
#             box_counts.append(count)     
#      # plot the box counts against the box sizes
#      plt.loglog(box_sizes, box_counts, 'o')
#      plt.xlabel('Box size (log scale)')
#      plt.ylabel('Number of boxes (log scale)')
#      plt.title('Box counting plot')
#      plt.show()
#      
# #     # compute the fractal dimension as the slope of the linear regression line
#      coeffs = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)
#      return coeffs[0]
# # 
# print(fractal_dimension(data['activity']))
# 
# =============================================================================
# =============================================================================

# =============================================================================
# # Load the dataset into a pandas dataframe
# #df = pd.read_csv('AllReadings.csv')
# df = newData.loc[((newData['Category']== 'Control'))]
# # Normalize the activity values using z-score normalization
# df['activity'] = (df['activity'] - df['activity'].mean()) / df['activity'].std()
# 
# # Remove 0 values
# #df = df[df['activity'] != 0]
# 
# # Remove outliers using the interquartile range method
# Q1 = df['activity'].quantile(0.25)
# Q3 = df['activity'].quantile(0.75)
# IQR = Q3 - Q1
# df = df[(df['activity'] >= Q1 - 1.5*IQR) & (df['activity'] <= Q3 + 1.5*IQR)]
# 
# # Create a boxplot of the normalized and filtered activity values
# plt.boxplot(df['activity'])
# plt.xlabel('Control')
# plt.show()
# 
# =============================================================================

# =============================================================================
# # separate data for each category
# control_data = newData[newData['Category'] == 'Control'].copy()
# depressive_data = newData[newData['Category'] == 'Depressive'].copy()
# schizophrenic_data = newData[newData['Category'] == 'Schizophrenic'].copy()
# 
# # normalize each dataframe using .loc accessor
# control_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] = (control_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] - control_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].mean()) / control_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].std()
# depressive_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] = (depressive_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] - depressive_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].mean()) / depressive_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].std()
# schizophrenic_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] = (schizophrenic_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']] - schizophrenic_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].mean()) / schizophrenic_data.loc[:, ['f.mean', 'f.sd', 'f.propZeros']].std()
# 
# # merge dataframes back together
# normalized_df = pd.concat([control_data, depressive_data, schizophrenic_data])
# 
# # print normalized dataframe
# print(normalized_df)
# ControlData = normalized_df.loc[((normalized_df['Category']== 'Control'))]
# plt.hist(ControlData['f.mean'], bins=100)
# plt.xlabel('Daily Activity of CONTROL Normalized')
# plt.ylabel('Frequency')
# plt.title('Histogram of Normalized Control Activity')
# plt.show()
# DepressionData = normalized_df.loc[((normalized_df['Category']== 'Depressive'))]
# plt.hist(DepressionData['f.mean'], bins=100)
# plt.xlabel('Dail Activity of DEPRESSIVE Normalized')
# plt.ylabel('Frequency')
# plt.title('Histogram of NormalizedDepression Activity')
# plt.show()
# SchizophreniaData = normalized_df.loc[((normalized_df['Category']== 'Schizophrenic'))]
# plt.hist(SchizophreniaData['f.mean'], bins=100)
# plt.xlabel('Daily Activity of SCHIZOPHRENIA Normalized')
# plt.ylabel('Frequency')
# plt.title('Histogram of Normalized Schizophrenic Activity')
# plt.show()
# 
# # Create a sample dataframe
# print(newData)
# =============================================================================



