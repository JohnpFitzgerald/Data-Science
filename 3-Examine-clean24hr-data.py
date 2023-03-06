# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:13:50 2023

@author: Jfitz
"""


import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# read the data from a CSV file
#data = pd.read_csv('CleanedActivityReadingsALL.csv')
#Data = pd.read_csv('24HourReturns.csv')
df = pd.read_csv('24HrAgg.csv')
df = pd.DataFrame(df)
# convert the "date" column to a datetime object
df['date'] = pd.to_datetime(df['date'])
#Data['timestamp'] = pd.to_datetime(Data['timestamp'])    
print(df.dtypes)
print(f"no of records:  {df.shape[0]}")
print(f"no of variables: {df.shape[1]}")


ControlData = df.loc[((df['Category']== 'Control'))]
plt.hist(ControlData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of CONTROL')
plt.ylabel('Frequency')
plt.title('Histogram of Control Activity daily averages')
plt.show()
DepressionData = df.loc[((df['Category']== 'Depressive'))]
plt.hist(DepressionData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of DEPRESSIVE')
plt.ylabel('Frequency')
plt.title('Histogram of Depression Activity daily averages')
plt.show()
SchizophreniaData = df.loc[((df['Category']== 'Schizophrenic'))]
plt.hist(SchizophreniaData['f.mean'], bins=50)
plt.xlabel('Daily average Activity of SCHIZOPHRENIA')
plt.ylabel('Frequency')
plt.title('Histogram of Schizophrenic Activity daily averages')
plt.show()



# calculate z-scores for activity data
z_scores = np.abs(stats.zscore(df['f.mean']))

# define threshold for outliers
lthreshold = 1
hthreshold = 300
#data = data['f.mean' > 10]
# filter out rows with z-score > threshold
#data = data[z_scores <= hthreshold]
#data = data[z_scores >= lthreshold]
df = df.loc[~((df['f.mean'] < 1))] 
df = df.loc[~((df['f.mean'] >300))] 
# normalize activity data using min-max scaling
df['zScore'] = (df['f.mean'] - df['f.mean'].min()) / (df['f.mean'].max() - df['f.mean'].min())

# print the normalized data
print(df)


# calculate mean, median, mode, and standard deviation
mean = np.mean(df['f.mean'])
median = np.median(df['f.mean'])
std_dev = np.std(df['f.mean'])

# calculate range
range = np.max(df['f.mean']) - np.min(df['f.mean'])

# calculate percentiles
percentiles = np.percentile(df['f.mean'], [25, 50, 75])

# print the results
print('Mean:', mean)
print('Median:', median)
#print('Mode:', mode)
print('Standard deviation:', std_dev)
print('Range:', range)
print('25th, 50th, and 75th percentiles:', percentiles)


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
 
print(entropy(df['f.mean']))

# =============================================================================
# def fractal_dimension(d):
#     # generate a range of box sizes
#     box_sizes = np.floor(np.logspace(0, np.log2(len(d)), num=20))
#     boxes = box_sizes.round().astype(int)
#     # count the number of boxes that contain at least one data point for each box size
#     box_counts = []
#     box_size = 0
#     for box_size in boxes:
#         if box_size > 0:
#             count = 0
#             for i in range(0, len(d), int(box_size)):
#                 if np.sum(d[i:i+int(box_size)]) > 0:
#                     count += 1
#             box_counts.append(count)
# 
#     # plot the box counts against the box sizes
#     plt.loglog(boxes, box_counts, 'o')
#     plt.xlabel('Box size (log scale)')
#     plt.ylabel('Number of boxes (log scale)')
#     plt.title('Box counting plot')
#     plt.show()
# 
#     # compute the fractal dimension as the slope of the linear regression line
#     coeffs = np.polyfit(np.log(boxes), np.log(box_counts), 1)
#     return coeffs[0]
# 
# d_rounded = df['f.mean'].round().astype(int)
# print(d_rounded)
# print(fractal_dimension(d_rounded.to_numpy()))
# =============================================================================

# do a line grapgh of z-scores on the average daily activity

df = df.loc[~((df['counter'] > 350))] 
grouped = df.groupby(['Category', 'counter'])['zScore'].mean().reset_index()
pivoted = grouped.pivot(index='counter', columns='Category', values='zScore')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

#plt.xticks(range(24), [f'{h:02d}' for h in range(24)])


plt.xlabel('z-Scores on daily averages')
plt.ylabel('Average Activity')
plt.title('z-Scores of the mean daily actvity by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('AverageActivityZScores.png', dpi=300)
plt.show()




