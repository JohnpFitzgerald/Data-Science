# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:44:27 2023

@author: fitzgeraldj
"""
import pandas as pd
import matplotlib.pyplot as plt


#
#    Standalone bit of script to produc a graph based on a 300 days of recorded
#    24hr data for the 3 test groupings
#
#
#



# read the data from a CSV file
#data = pd.read_csv('CleanedActivityReadingsALL.csv')
data = pd.read_csv('24HrAgg.csv')
df = pd.DataFrame(data)
# convert the "date" column to a datetime object
df['date'] = pd.to_datetime(df['date'])
#df['timestamp'] = pd.to_datetime(df['timestamp'])
fig, ax = plt.subplots()

# Loop through each category and plot the mean values against the counter
for category in df['Category'].unique():
    data = df[df['Category'] == category]
    ax.plot(data['counter'], data['f.mean'], label=category)

# Set the x and y axis labels
ax.set_xlabel('Days')
ax.set_ylabel('Daily Mean')

# Add a legend to the plot
# Add a legend to the plot
ax.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('DailyMeanActivity.png', dpi=300)
 # Show the plot
plt.show() 