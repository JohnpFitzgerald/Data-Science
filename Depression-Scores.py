# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:33:17 2023

@author: Jfitz
"""
import pandas as pd

# read the data from a CSV file
data = pd.read_csv('Depression-Scores.csv')
print(data)

#data['count'] = data.groupby(['gender','age'])['number'].transform('count')


data['id'] = data['number'].str[:5]


maskCondition = (data['id'] == 'condi')
maskControl = (data['id'] == 'contr')

rows_cond = data[maskCondition]
rows_contr = data[maskControl]

print(data)

result = data.groupby(['gender','age']).filter(lambda x: len(x) > 1)['id'].tolist()

print(result)

