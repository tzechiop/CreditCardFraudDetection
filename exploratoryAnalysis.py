# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:23:51 2017

@author: thasegawa
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# File parameters
fname = 'creditcard.csv'

# Read file
df = pd.read_csv(fname)

outdir = r'C:\Users\thasegawa\Documents\Kaggle\Credit Card Fraud Detection\Exploratory Results'
# Create summary statistics
pd.options.display.max_columns = 999
out = df.describe()
print(out)
out.to_csv(os.path.join(outdir, 'DescribeFeatures.csv'))

# View distribution of Class
out = df['Class'].value_counts()
print(out)
out.to_csv(os.path.join(outdir, 'ValueCounts.csv'))

# See if there are null values
out = df.isnull().sum()
print(out)
out.to_csv(os.path.join(outdir, 'NullCounts.csv'))

# View distribution of each variable based on class
xcols = df.columns.values[:-1]
num_xcols = len(xcols)
for i, xcol in enumerate(xcols):
    fig, ax = plt.subplots()
    sns.distplot(df[xcol].loc[df['Class'] == 0], label = 'Normal')
    sns.distplot(df[xcol].loc[df['Class'] == 1], label = 'Fraud')
    ax.set_xlabel('')
    ax.set_ylabel('Histogram of feature: %s' % xcol)
    plt.show()
    ax.legend()
    outpath = os.path.join(outdir, 'Histogram_{0:0>2d}_{1}.pdf'.format(i, xcol))
    fig.savefig(outpath, bbox_inches = 'tight')
    
# Create features to show probability of fraud/normal based on 100 bins
