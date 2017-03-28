# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:10:46 2017

@author: thasegawa
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Parameters
test_size = 0.25            # Split for train + validation / test datasets
tv_test_size = 0            # Split for train / validation datasets
k = 10                      # Number of splits

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

# Split data to create train + validation and test datasets
X_maj_tv, X_maj_test, y_maj_tv, y_maj_test = train_test_split(X_maj, y_maj, test_size = test_size, random_state = 0)
X_min_tv, X_min_test, y_min_tv, y_min_test = train_test_split(X_min, y_min, test_size = test_size, random_state = 0)

# Output test datasets
print('Creating test dataset...')
X_test_all = pd.concat([X_maj_test, X_min_test], axis = 0)
y_test_all = pd.concat([y_maj_test, y_min_test], axis = 0)
pd.DataFrame(y_test_all).to_csv('Dataset_test_y.csv')
X_test_all.to_csv('Dataset_test_X.csv')

# Split data
rs = KFold(n_splits = k, random_state = 0)
majidx_list = []
for idx in rs.split(X_maj_tv):
    majidx_list.append(idx)
    
minidx_list = []
for idx in rs.split(X_min_tv):
    minidx_list.append(idx)
    
for index, ((majidx_train, majidx_valid), (minidx_train, minidx_valid)) in enumerate(zip(majidx_list, minidx_list)):
    print('Creating {0:0>2d}th-fold dataset...'.format(index + 1))
    y_train = pd.concat([y_maj_tv.iloc[majidx_train], y_min_tv.iloc[minidx_train]], axis = 0)
    X_train = pd.concat([X_maj_tv.iloc[majidx_train], X_min_tv.iloc[minidx_train]], axis = 0)
    pd.DataFrame(y_train).to_csv('Dataset{0:0>2d}_y_train.csv'.format(index))
    X_train.to_csv('Dataset{0:0>2d}_X_train.csv'.format(index))

    y_test = pd.concat([y_maj_tv.iloc[majidx_valid], y_min_tv.iloc[minidx_valid]], axis = 0)
    X_test = pd.concat([X_maj_tv.iloc[majidx_valid], X_min_tv.iloc[minidx_valid]], axis = 0)
    pd.DataFrame(y_test).to_csv('Dataset{0:0>2d}_y_valid.csv'.format(index))
    X_test.to_csv('Dataset{0:0>2d}_X_valid.csv'.format(index))

# Create a dataset that includes all testing and validation training datasets
print('Creating a summation of all testing and validation datasets...')
majidx_train_all = []
minidx_train_all = []
for (majidx_train, majidx_test), (minidx_train, minidx_test) in zip(majidx_list[:10], minidx_list[:10]):
    majidx_train_all += list(majidx_train)
    minidx_train_all += list(minidx_train)

y_train_all = pd.concat([y_maj_tv.iloc[majidx_train_all], y_min_tv.iloc[minidx_train_all]], axis = 0)    
X_train_all = pd.concat([X_maj_tv.iloc[majidx_train_all], X_min_tv.iloc[minidx_train_all]], axis = 0)
pd.DataFrame(y_train_all).to_csv('Dataset_train_all_y.csv')
X_train_all.to_csv('Dataset_train_all_X.csv')
