# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:23:51 2017

@author: thasegawa
"""

import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('Dataset_train_all_X.csv', index_col = 0)
y = pd.read_csv('Dataset_train_all_y.csv', index_col = 0)

xcols = X.columns.values

# Determine bins in which there are more fradulent transactions
bins_list = []
morefraud_list = []
for xcol in xcols:
    freq_norm, bins_norm, fig = plt.hist(X[y == 0], bins = 100)
    freq_fraud, bins_fraud, fig = plt.hist(X[y == 1], bins = bins_norm)
    break