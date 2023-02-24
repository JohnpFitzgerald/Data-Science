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
