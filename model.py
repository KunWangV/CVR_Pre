# coding: utf-8
# pylint: disable=C0103, C0111
import scipy as sp
import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.utils import shuffle
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

from data import *
from feature import *


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(
        sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    print '-------------logloss-----------', ll
    return ll


def LGB(X, y, pred_x):
    pass


def RF(x, y, pred_x):
    """
    随机森林，计算特征重要性
    """
    print '----rf-----'
    print x.shape
    print pred_x.shape

    posnum = y[y == 1].shape[0]
    negnum = y[y == 0].shape[0]
    print 'pos:', posnum, ' neg:', negnum
    weight = float(posnum) / (posnum + negnum)
    print 'weight:', weight

    xtrain, xvalid, ytrain, yvalid = train_test_split(
        x, y, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=6,
                                 random_state=500,
                                 class_weight={1: weight},
                                 n_jobs=8)\
        .fit(xtrain, ytrain)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print indices
    print importances[indices]

    imp = pd.DataFrame(importances, columns=['imp'])
    imp.to_csv('importances.csv', index=False)

    test_pred = clf.predict_proba(xvalid)
    logloss(yvalid, test_pred[:, 1])
    pred = clf.predict_proba(pred_x)
    return pred[:, 1], indices


def XGB(X, y, pred_x):
    pass


def NN(X, y):
    pass


def main():
    train_x, train_y, test_x, inst = load_feature(from_file=True)
    RF(train_x, train_y, test_x)


if __name__ == '__main__':
    main()