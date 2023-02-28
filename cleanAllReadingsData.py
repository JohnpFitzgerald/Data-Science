# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:55:38 2023

@author: fitzgeraldj
"""

import pandas as pd
import matplotlib.pyplot as plt

# read the data from a CSV file
data = pd.read_csv('AllReadings.csv')

# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])
# plot the data

groupings = data.groupby('date').mean()['activity']

groupings.plot()

# set the x-axis label
plt.xticks(rotation=45)


# set the title of the plot
plt.title('Reading by Date')

# display the plot
plt.show()


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
    
data['counter'] = data.groupby('Category').cumcount() + 1

data.to_csv('C:/mtu/project/CleanedActivityReadingsALL.csv', index=False)    
# =============================================================================
# =============================================================================
# groupings = df.groupby(['category', pd.Grouper(key='date', freq='D')]).mean()
# # 
# groupings.plot()
# # 
# # # set the x-axis label
# plt.xticks(rotation=45)
# # 
# # 
# # # set the title of the plot
# plt.title('Reading by Category')
# # 
# # # display the plot
# plt.show()
# =============================================================================
# =============================================================================






    