# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:36:36 2017

Documentation for Models
SVM: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
Random Forest: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

Documentation for Metrics
log_loss: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
@author: thasegawa
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from datetime import datetime

index = 0
fname_X_train = 'BootstrapDataset{0:0>2d}_X_train.csv'.format(index)
fname_y_train = 'BootstrapDataset{0:0>2d}_y_train.csv'.format(index)
fname_X_test = 'BootstrapDataset{0:0>2d}_X_valid.csv'.format(index)
fname_y_test = 'BootstrapDataset{0:0>2d}_y_valid.csv'.format(index)

# Read input data
print('Reading input data...')
X_train = pd.read_csv(fname_X_train, index_col = 0)
y_train = np.ravel(pd.read_csv(fname_y_train, index_col = 0, header = None))
X_test = pd.read_csv(fname_X_test, index_col = 0)
y_test = np.ravel(pd.read_csv(fname_y_test, index_col = 0, header = None))

# ===========================================================================
# Run SVC
print('Running SVC...')
startTime = datetime.now()

clf_SVC = svm.SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
y_predict = clf_SVC.predict(X_test)
fpr_SVC, tpr_SVC, thr_SVC = roc_curve(y_test, y_predict)
pr_SVC, rec_SVC, thr_SVC = precision_recall_curve(y_test, y_predict)

delta_SVC = datetime.now() - startTime
i = list(thr_SVC).index(1)
print('\tPrecision: {0}'.format(pr_SVC[i]))
print('\tRecall: {0}'.format(rec_SVC[i]))
print('\tTime: {0}'.format(delta_SVC))

# Run Random Forest
print('Running Random Forest...')
startTime = datetime.now()

clf_RF = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
y_predict = clf_RF.predict(X_test)
fpr_RF, tpr_RF, thr_RF = roc_curve(y_test, y_predict)
pr_RF, rec_RF, thr_RF = precision_recall_curve(y_test, y_predict)

delta_RF = datetime.now() - startTime
i = list(thr_RF).index(1)
print('\tPrecision: {0}'.format(pr_RF[i]))
print('\tRecall: {0}'.format(rec_RF[i]))
print('\tTime: {0}'.format(delta_RF))

# Run Gradient Boosting
print('Running Gradient Boosting...')
startTime = datetime.now()

clf_GB = GradientBoostingClassifier(n_estimators = 100).fit(X_train, y_train)
y_predict = clf_GB.predict(X_test)
fpr_GB, tpr_GB, thr_GB = roc_curve(y_test, y_predict)
pr_GB, rec_GB, thr_GB = precision_recall_curve(y_test, y_predict)

delta_GB = datetime.now() - startTime
i = list(thr_GB).index(1)
print('\tPrecision: {0}'.format(pr_GB[i]))
print('\tRecall: {0}'.format(rec_GB[i]))
print('\tTime: {0}'.format(delta_GB))
