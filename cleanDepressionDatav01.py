# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:55:38 2023

@author: fitzgeraldj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the data from a CSV file
data = pd.read_csv('DepressionReadings.csv')

# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
#data['day'] = pd.to_datetime(data['date'],format='%d/%m/%Y').dt.day
data['timestamp'] = pd.to_datetime(data['timestamp'], format= '%Y/%m/%d %H:%M')
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['ddmm'] = data['timestamp'].dt.strftime('%d%m')
#data['no'] = data.groupby('id').cumcount()+1
data.loc[:, 'no'] = data.groupby('id').cumcount() + 1
data.fillna(0, inplace=True)




#extract2225 = data.query("id == 'condition_22' or id == 'control_25'")
extract22 = data.query("id == 'condition_22'")
extract25 = data.query("id == 'control_25'")
#extract19.to_csv('C:/mtu/project/patient16-control28.csv', index=False)
#extract22.loc[:, 'no'] = extract22.groupby('id').cumcount() + 1
#extract25.loc[:, 'no'] = extract25.groupby('id').cumcount() + 1



daily_mean22 = extract22.groupby(['no','id'])['activity'].mean()
daily_mean25 = extract25.groupby(['no','id'])['activity'].mean()
daily_mean22 = daily_mean22.unstack()
daily_mean25 = daily_mean25.unstack()

   
plt.plot(daily_mean25.index, daily_mean25.values, label="Control 25")
plt.plot(daily_mean22.index, daily_mean22.values, label="Patient 22")

plt.xlabel('Minutes recorded per person')
plt.ylabel('Average activity rate')
plt.title('Daily average activity rate Patient 22 v Control 25')
plt.legend()
plt.show()
# =============================================================================

extract22 = data.query("id == 'condition_22'")
extract25 = data.query("id == 'control_25'")
Patient22 = extract22.groupby(['date'])['activity'].mean()
Control25 = extract25.groupby(['date'])['activity'].mean()

activ = [Patient22,Control25]
#print(activ22)
#boxPlot2522 = []

fig, ax = plt.subplots()

ax.boxplot(activ)


ax.set_xticklabels(['Patient','Control'])
ax.set_ylabel('daily average activity')
#ax.set_xlabel('Patient 22 v Control 25')
ax.set_title("Box plot of daily average activities Patient 22 v Control 25")
plt.show() 
# =============================================================================
#Control 9 Pateient 20
extract09 = data.query("id == 'control_9'")
extract20 = data.query("id == 'condition_20'")

#extract09.loc[:, 'no'] = extract09.groupby('id').cumcount() + 1
#extract20.loc[:, 'no'] = extract20.groupby('id').cumcount() + 1

daily_mean09 = extract09.groupby(['no','id'])['activity'].mean()
daily_mean20 = extract20.groupby(['no','id'])['activity'].mean()

daily_mean09 = daily_mean09.unstack()
daily_mean20 = daily_mean20.unstack()

#print(daily_mean09)

plt.plot(daily_mean09.index, daily_mean09.values, label="Control 09")   
plt.plot(daily_mean20.index, daily_mean20.values, label="Patient 20")


plt.xlabel('Minutes recorded per person')
plt.ylabel('Average activity rate')
plt.title('Daily average activity rate Patient 20 v Control 09')
plt.legend()
plt.show()





Patient20 = extract20.groupby(['date'])['activity'].mean()
Control09 = extract09.groupby(['date'])['activity'].mean()

active = [Patient20,Control09]

fig, ax = plt.subplots()

ax.boxplot(active)

ax.set_xticklabels(['Patient','Control'])
ax.set_ylabel('daily average activity')
#ax.set_xlabel('Patient 22 v Control 25')
ax.set_title("Box plot of daily average activities Patient 20 v Control 9")
plt.show() 







act_mean = data['activity'].mean()        # Calculate the mean age
act_median = data['activity'].median()    # Calculate the median age
act_min = data['activity'].min()          # Calculate the minimum age
act_max = data['activity'].max()          # Calculate the maximum age
act_var = data['activity'].var()          # Calculate the variance of ages
act_std = data['activity'].std()          # Calculate the standard deviation of ages
act_count = data['activity'].count()      # Calculate the number of non-missing age values

print("Activity Stats:  MEAN: "+str(act_mean)," MEDIAN: "+str(act_median))
print("Activity Stats:  MIN: "+str(act_min)," MAX: "+str(act_max))
print("Activity Stats:  Variance: "+str(act_var)," Standard Deviation: "+str(act_std))
print("Activity Stats:  COUNT: "+str(act_count))


data['id2'] = data['id'].str[:5]
data['date'] = data['date'].astype(str)
#maskCondition = (data['id'] == 'condi')
#data['date_id'] = data['date'].astype(str) + '_' + data['id2'].astype(str)


newData = pd.pivot_table(data, values='activity', index='date', columns=(data['id2']=='condi'))

sns.heatmap(newData)
