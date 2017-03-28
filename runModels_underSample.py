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
fname_X_train = 'Dataset{0:0>2d}_X_train.csv'.format(index)
fname_y_train = 'Dataset{0:0>2d}_y_train.csv'.format(index)
fname_X_test = 'Dataset{0:0>2d}_X_valid.csv'.format(index)
fname_y_test = 'Dataset{0:0>2d}_y_valid.csv'.format(index)

# Read input data
print('Reading input data...')
X_train = pd.read_csv(fname_X_train, index_col = 0)
y_train = np.ravel(pd.read_csv(fname_y_train, index_col = 0))
X_test = pd.read_csv(fname_X_test, index_col = 0)
y_test = np.ravel(pd.read_csv(fname_y_test, index_col = 0))

# Run SVC
print('Running SVC...')
startTime = datetime.now()

clf_SVC = svm.SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
y_predict = clf_SVC.predict(X_test)
fpr_SVC, tpr_SVC, _ = roc_curve(y_test, y_predict)
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
pr_RF, rec_RF, thr_RF= precision_recall_curve(y_test, y_predict, pos_label = 1)

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
fpr_GB, tpr_GB, _ = roc_curve(y_test, y_predict)
pr_GB, rec_GB, thr_GB = precision_recall_curve(y_test, y_predict)

delta_GB = datetime.now() - startTime
i = list(thr_GB).index(1)
print('\tPrecision: {0}'.format(pr_GB[i]))
print('\tRecall: {0}'.format(rec_GB[i]))
print('\tTime: {0}'.format(delta_GB))

# ===========================================================================
# Repeat process with an oversampled balanced dataset
print('Running models with oversampled balanced dataset...')
# Count number of fraud and non-fraud data points
min_idx = y_train == 1

X_train_min = X_train[min_idx]
X_train_maj = X_train[~min_idx]
y_train_min = pd.DataFrame(y_train[min_idx])
y_train_maj = pd.DataFrame(y_train[~min_idx])
num_min = X_train_min.shape[0]
num_maj = X_train_maj.shape[0]

# sample minority dataset with replacement for num_maj times
X_train_min_over = X_train_min.sample(n = num_maj, replace = True)
y_train_min_over = y_train_min.sample(n = num_maj, replace = True)

# Recombine datasets
X_train_over = pd.concat([X_train_maj, X_train_min_over], axis = 0)
y_train_over = np.ravel(pd.concat([y_train_maj, y_train_min_over], axis = 0))

# Run SVC
print('Running SVC...')
startTime = datetime.now()

clf_SVC = svm.SVC(kernel = 'linear', C = 1).fit(X_train_over, y_train_over)
y_predict = clf_SVC.predict(X_test)
fpr_SVC, tpr_SVC, thr_SVC = roc_curve(y_test, y_predict)

delta_SVC = datetime.now() - startTime
i = list(thr_SVC).index(1)
print('\tPrecision: {0}'.format(pr_SVC[i]))
print('\tRecall: {0}'.format(rec_SVC[i]))
print('\tTime: {0}'.format(delta_SVC))

# Run Random Forest
print('Running Random Forest...')
startTime = datetime.now()

clf_RF = RandomForestClassifier(n_estimators = 100).fit(X_train_over, y_train_over)
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

clf_GB = GradientBoostingClassifier(n_estimators = 100).fit(X_train_over, y_train_over)
y_predict = clf_GB.predict(X_test)
fpr_GB, tpr_GB, thr_GB = roc_curve(y_test, y_predict)
pr_GB, rec_GB, thr_GB = precision_recall_curve(y_test, y_predict)

delta_GB = datetime.now() - startTime
i = list(thr_GB).index(1)
print('\tPrecision: {0}'.format(pr_GB[i]))
print('\tRecall: {0}'.format(rec_GB[i]))
print('\tTime: {0}'.format(delta_GB))