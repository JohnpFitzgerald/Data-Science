# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:06:04 2023

@author: fitzgeraldj
"""

import pandas as pd
import os



def mergeLoop(files):
# Create a DataFrame to store the merged data
  mergedData = pd.DataFrame()    
# Loop through all CSV files in the named directory and concatenate into the new DataFrame
# and apply the filename as a new variable
  for file in os.listdir(files):
    if file.endswith('.csv'):
        # Read the CSV file into a DataFrame and add a new column to identify the file
        data = pd.read_csv(os.path.join(files, file))
        data['id'] = file.split('.')[0]  # Add a new column with the file name as identifier
        
        # Append the DataFrame to the merged_data DataFrame
        mergedData = pd.concat([mergedData, data], ignore_index=True)    
  return mergedData


def mergeDepression():
#merge the DEPRESSION files along with the control group files and apply a
#new column to identify patient/Control:
# Set the path to the directory containing the CSV files
   path = 'C:/mtu/project/depMerge/'    
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/DepressionReadings.csv', index=False)

   
   
def mergeSchizophrenia():
#merge the SCHIZOPHRENIA files along with the CONTROL group files
# and apply a new variable to identify Condition or Control:
# Set the path to the directory containing the CSV files
   path = 'C:/mtu/project/schMerge/'
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/SchizophreniaReadings.csv', index=False) 
   
mergeSchizophrenia() 
mergeDepression()   

