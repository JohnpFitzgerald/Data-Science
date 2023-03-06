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
import seaborn as sns
# =============================================================================
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
# Calculate the minute of the day
data['minute'] = data['timestamp'].dt.minute + data['hour'] * 60
#aggregate date and hour and include data for 24 hour period only
aggr = data.groupby(['date','hour']).agg({'activity': 'sum'}).reset_index()
aggr = aggr[(aggr['hour'] >= 0) & (aggr['hour'] <= 23)]
counted = aggr.groupby('date').agg({'hour' : 'count'}).reset_index()
counted = counted[counted['hour'] == 24]

final = pd.merge(data, counted[['date']], on='date', how='inner')
#examine the data
print(final.head())
 



#final['datetime'] = pd.to_datetime(final['date'] + ' ' + final['hour'], format='%d/%m/%Y %H:%M')
counts = final.groupby(['id', 'date']).count()
valid_groups = counts[counts['activity'] == 1440].reset_index()[['id', 'date']]
final = final.merge(valid_groups, on=['id', 'date'])
#final.drop('timestamp', axis=1, inplace=True)
def newId(idVal):
    if idVal[:5] == 'condi':
        return 'Schizophrenic'
    elif idVal[:5] == 'patie':
        return 'Depressive'
    elif idVal[:5] == 'contr':
        return 'Control'
    else:
        return '*UNKNOWN*'
  
final['Category'] = final['id'].apply(newId)

if '*UNKNOWN*' in final['Category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")        


 
num_records = len(final)
print(f"Number of records in dataframe: {num_records}")
 
if num_records % 1440 == 0:
   print("Number of records is divisible by 1440 with no remainder")
else:
   print("Number of records is not divisible by 1440 with no remainder")
 
final.to_csv('C:/mtu/project/24HourReturns.csv', index=False) 


 
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

if '*UNKNOWN*' in newData['Category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")   
 
newData['counter'] = newData.groupby('Category').cumcount() + 1

print(newData)    
 

print("***  All 3 groups Baseline input file created for 24 hr of data only ***")
# =============================================================================
# plOTS
#-----------------------------------------------------------------------------
newData.to_csv('C:/mtu/project/24HrAgg.csv', index=False)

grouped = final.groupby(['Category', 'hour'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='hour', columns='Category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

plt.xticks(range(24), [f'{h:02d}' for h in range(24)])


plt.xlabel(' 24 hour period hourly')
plt.ylabel('Average Activity')
plt.title('Average Activity midnight to midnight per hour by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('AverageActivityPerHour.png', dpi=300)
plt.show()




grouped = final.groupby(['Category', 'minute'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='minute', columns='Category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

#plt.xticks(range(1440), [f'{h:0}' for h in range(1440)])


plt.xlabel(' 24 hour period in minutes')
plt.ylabel('Average Activity')
plt.title('Average Activity per minute by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('AverageActivityPerMinute.png', dpi=300)
plt.show()

sns.barplot(x='Category', y='activity', data=final)

