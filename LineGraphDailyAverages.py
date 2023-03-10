# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:18:19 2023

@author: fitzgeraldj
"""

import pandas as pd
import matplotlib.pyplot as plt

# read the data from a CSV file
#data = pd.read_csv('CleanedActivityReadingsALL.csv')
data = pd.read_csv('24HrAgg.csv')
df = pd.DataFrame(data)
# convert the "date" column to a datetime object
df['date'] = pd.to_datetime(df['date'])
#df['timestamp'] = pd.to_datetime(df['timestamp'])




# =============================================================================
# # group by category and date, and calculate daily mean for each category
# daily_mean = df.groupby(['Category', 'counter']).mean().reset_index()
# 
# # create line graph with multiple lines, one for each category
# fig, ax = plt.subplots(figsize=(10, 6))
# for category in daily_mean['Category'].unique():
#     data = daily_mean[daily_mean['Category'] == category]
#     ax.plot(data['counter'], data['activity'], label=category)
# 
# # set graph properties
# ax.set_xlabel('Days')
# ax.set_ylabel('Daily Mean')
# ax.set_title('Daily Mean by Category')
# ax.legend()
# 
# # display graph
# plt.show()
# =============================================================================

# create sample dataframe


# =============================================================================
# # group by category and timestamp, calculate mean activity
# grouped = df.groupby(['Category', pd.Grouper(key='date', freq='D')]).mean()
# 
# # plot line graph
# fig, ax = plt.subplots(figsize=(10,6))
# 
# for category in df['Category'].unique():
#     data = grouped.loc[category]['activity']
#     ax.plot(data.index.day, data.values, label=category)
# 
# ax.set_xlabel('Day')
# ax.set_ylabel('Mean Activity')
# ax.legend()
# plt.show()
# =============================================================================

# create example data


# =============================================================================
# 
# # set up plot
# fig, ax = plt.subplots()
# 
# # loop through each category and plot a line
# for category in df['Category'].unique():
#     df_category = df[df['Category'] == category]
#     ax.plot(df_category['counter'], df_category['activity'], label=category)
# 
# 
# # add legend and axis labels
# ax.legend()
# ax.set_xlabel('Day')
# ax.set_ylabel('Value')
# 
# # display plot
# plt.show()
# =============================================================================



# Create a figure and axis object
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
# Show the plot
plt.show()
plt.savefig('DailyMeanActivity.png', dpi=300)
# =============================================================================
# # Loop through each category and plot the mean values against the counter
# for category in df['Category'].unique():
#     data = df[df['Category'] == category]
#     ax.plot(data['counter'], data['f.sd'], label=category)
# 
# # Set the x and y axis labels
# ax.set_xlabel('Days')
# ax.set_ylabel('Daily Standard Deviation')
# 
# # Add a legend to the plot
# ax.legend()
# 
# # Show the plot
# plt.show()
# 
# # Loop through each category and plot the mean values against the counter
# for category in df['Category'].unique():
#     data = df[df['Category'] == category]
#     ax.plot(data['counter'], data['f.propZeros'], label=category)
# 
# # Set the x and y axis labels
# ax.set_xlabel('Days')
# ax.set_ylabel('Daily Standard Deviation')
# 
# # Add a legend to the plot
# ax.legend()
# # Set the size of the graph
# plt.gcf().set_size_inches(12, 6)
# 
# # Save the graph to a file
# plt.savefig('DailyMeanActivity.png', dpi=300)
# # Show the plot
# plt.show()
# 
# =============================================================================







