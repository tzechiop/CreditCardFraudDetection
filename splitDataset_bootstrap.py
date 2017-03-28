# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:10:46 2017

@author: thasegawa
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Parameters
test_size = 0.25              # Test size for test dataset (with same ratio of minority/majority as original dataset)
num_bootstrap = 10
ratio_min = 0.8               # Ratio between num_min and size of bootstrap for minority data
ratio_maj = 0.8               # Ratio between num_min and size of bootstrap for majority data

# Read data
data = pd.read_csv('creditcard.csv')
xcols = data.columns.values[:-1]
minidx = (data['Class'] == 1)
X_min = data[xcols].loc[minidx]
X_maj = data[xcols].loc[~minidx]
y_min = data['Class'].loc[minidx]
y_maj = data['Class'].loc[~minidx]

# split data to train+validation / test data
X_tv_min, X_test_min, y_tv_min, y_test_min = train_test_split(X_min, y_min, test_size = test_size)
X_tv_maj, X_test_maj, y_tv_maj, y_test_maj = train_test_split(X_maj, y_maj, test_size = test_size)

# Output test data
X_test = pd.concat([X_test_min, X_test_maj], axis = 0)
y_test = pd.concat([y_test_min, y_test_maj], axis = 0)
X_test.to_csv('BootstrapDataset_Test_X.csv')
y_test.to_csv('BootstrapDataset_Test_y.csv')


# Get number of minority and majority data
num_min = y_tv_min.shape[0]
num_maj = y_tv_maj.shape[0]

for index in range(num_bootstrap):
    print('Creating {0:0>2d}th bootstrap dataset...'.format(index + 1))
    # Get bootstrap samples of data, which is the training data
    num_bs_min = int(round(num_min*ratio_min))
    num_bs_maj = int(round(num_min*ratio_maj))
    
    X_train_min = X_tv_min.sample(n = num_bs_min)
    X_train_maj = X_tv_maj.sample(n = num_bs_maj)
    
    minidx_train = X_train_min.index
    majidx_train = X_train_maj.index
    
    y_train_min = y_tv_min[y_tv_min.index.isin(minidx_train)]
    y_train_maj = y_tv_maj[y_tv_maj.index.isin(majidx_train)]
    
    # Get all other data, which is the validation data
    minidx_train_TF = X_tv_min.index.isin(minidx_train)
    majidx_train_TF = X_tv_maj.index.isin(majidx_train)
    
    X_valid_min = X_tv_min[~minidx_train_TF]
    X_valid_maj = X_tv_maj[~majidx_train_TF]
    y_valid_min = y_tv_min[~minidx_train_TF]
    y_valid_maj = y_tv_maj[~majidx_train_TF]
    
    # Concatenate majority and minority data and output
    X_train = pd.concat([X_train_min, X_train_maj], axis = 0)
    y_train = pd.concat([y_train_min, y_train_maj], axis = 0)
    X_valid = pd.concat([X_valid_min, X_valid_maj], axis = 0)
    y_valid = pd.concat([y_valid_min, y_valid_maj], axis = 0)
    
    X_train.to_csv('BootstrapDataset{0:0>2d}_X_train.csv'.format(index))
    y_train.to_csv('BootstrapDataset{0:0>2d}_y_train.csv'.format(index))
    X_valid.to_csv('BootstrapDataset{0:0>2d}_X_valid.csv'.format(index))
    y_valid.to_csv('BootstrapDataset{0:0>2d}_y_valid.csv'.format(index))