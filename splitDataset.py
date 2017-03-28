# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:10:46 2017

@author: thasegawa
"""

import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Read data
data = pd.read_csv('creditcard.csv')

# Determine x columns and split data to x and y
y = data['Class']
maj_idx = y == 0
min_idx = y == 1
y_maj = y[maj_idx]
y_min = y[min_idx]
xcols = data.columns.values[:-1]
X = data[xcols]
X_maj = data[xcols].loc[maj_idx]
X_min = data[xcols].loc[min_idx]

# Split data
rs = ShuffleSplit(n_splits = 11, test_size = 0.25, random_state = 0)
majidx_list = []
for idx in rs.split(X_maj):
    majidx_list.append(idx)
    
minidx_list = []
for idx in rs.split(X_min):
    minidx_list.append(idx)
    
for index, ((majidx_train, majidx_test), (minidx_train, minidx_test)) in enumerate(zip(majidx_list, minidx_list)):
    y_train = pd.concat([y_maj.iloc[majidx_train], y_min.iloc[minidx_train]], axis = 0)
    X_train = pd.concat([X_maj.iloc[majidx_train], X_min.iloc[minidx_train]], axis = 0)
    y_test = pd.concat([y_maj.iloc[majidx_test], y_min.iloc[minidx_test]], axis = 0)
    X_test = pd.concat([X_maj.iloc[majidx_test], X_min.iloc[minidx_test]], axis = 0)
    
    pd.DataFrame(y_train).to_csv('Dataset{0:0>2d}_y_train.csv'.format(index))
    X_train.to_csv('Dataset{0:0>2d}_X_train.csv'.format(index))
    pd.DataFrame(y_test).to_csv('Dataset{0:0>2d}_y_test.csv'.format(index))
    X_test.to_csv('Dataset{0:0>2d}_X_test.csv'.format(index))

# Create a dataset that includes the first 10 training datasets
majidx_train_all = []
minidx_train_all = []
for (majidx_train, majidx_test), (minidx_train, minidx_test) in zip(majidx_list[:10], minidx_list[:10]):
    majidx_train_all += list(majidx_train)
    minidx_train_all += list(minidx_train)

y_train_all = pd.concat([y_maj.iloc[majidx_train_all], y_min.iloc[minidx_train_all]], axis = 0)    
X_train_all = pd.concat([X_maj.iloc[majidx_train_all], X_min.iloc[minidx_train_all]], axis = 0)
pd.DataFrame(y_train_all).to_csv('Dataset_train_all_y.csv')
X_train_all.to_csv('Dataset_train_all_X.csv')