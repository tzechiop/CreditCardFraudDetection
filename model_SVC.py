# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:36:36 2017

@author: thasegawa
"""

from sklearn import svm
import pandas as pd


def runSVC(X_train, y_train, X_test, y_test, kernel = 'linear'):
    clf = svm.SVC(kernel = kernel, C = 1).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
return score