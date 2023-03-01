# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:30:08 2023

@author: Jfitz
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#3 files merged 
file = 'AllReadings.csv'
data = pd.read_csv(file)
# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])    
#examine data
print(data.dtypes)
print(f"no of records:  {data.shape[0]}")
print(f"no of variables: {data.shape[1]}")





#get hour value
data['hour'] = data['timestamp'].dt.hour

#aggregate date and hour and include data for 24 hour period only
aggr = data.groupby(['date','hour']).agg({'activity': 'sum'}).reset_index()
aggr = aggr[(aggr['hour'] >= 0) & (aggr['hour'] <= 23)]
counted = aggr.groupby('date').agg({'hour' : 'count'}).reset_index()
counted = counted[counted['hour'] == 24]

final = pd.merge(data, counted[['date']], on='date', how='inner')
#examine the data
print(final.head())

final.to_csv('C:/mtu/project/Aggregated24Hour.csv', index=False) 

# Convert the "timestamp" column to a datetime format
final['timestamp'] = pd.to_datetime(final['timestamp'], format='%d/%m/%Y %H:%M')

# Set the "timestamp" column as the index
final.set_index('timestamp', inplace=True)

# Resample the DataFrame at a minute-level frequency
df_resampled = final.resample('T').count()

# Check if the count for each day is equal to 1440
completeDay = (df_resampled.groupby(pd.Grouper(freq='D')).sum() == 1440).all(axis=1)


# Drop the incomplete days
#days = final.loc[completeDay.values]

completeDay.to_csv('C:/mtu/project/24HoursONLY.csv', index=False) 

grouped = final.groupby(['id','date'])
newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
newData = newData.reset_index()

newData.columns = ['id','date','f.mean', 'f.sd', 'f.propZeros']

newData['class1'] = newData['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
newData = newData[['id','class1','date','f.mean','f.sd','f.propZeros']]
newData = newData.loc[~((newData['f.mean'] == 0
                         ) & (newData['f.sd'] == 0))]
newData = newData.loc[~((newData['f.propZeros'] == 0))] 


def newId(idVal):
    if idVal[:5] == 'condi':
        return 'Schizophrenic'
    elif idVal[:5] == 'patie':
        return 'Depressive'
    elif idVal[:5] == 'contr':
        return 'Control'
    else:
        return '*UNKNOWN*'
    
newData['Category'] = newData['id'].apply(newId)

print(newData)    

if '*UNKNOWN*' in newData['Category'].values:
    print("unknowns found") 
print("***  All 3 groups Baseline input file created for 24 hr of data only ***")
newData['counter'] = newData.groupby('Category').cumcount() + 1
newData.to_csv('C:/mtu/project/24Hr-Aggregated-features.csv', index=False)

fig, ax = plt.subplots()

# Loop through each category and plot the mean values against the counter
for category in newData['Category'].unique():
    data = newData[newData['Category'] == category]
    ax.plot(data['counter'], data['f.mean'], label=category)

# Set the x and y axis labels
ax.set_xlabel('Days')
ax.set_ylabel('Daily Mean')

# Add a legend to the plot
# Add a legend to the plot
ax.legend()
plt.gcf().set_size_inches(12, 6)
# Show the plot
plt.show()
plt.savefig('DailyMeanActivity.png', dpi=300)
    