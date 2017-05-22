# coding: utf-8
# pylint: disable=C0103, C0111,C0326
import scipy as sp
# import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.utils import shuffle
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from data import *
from feature import *
import os

submit_flag=True

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(
        sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    print '-------------logloss-----------', ll
    return ll


def feature_select(x, y, pre_x, rate=0.2):
    if not os.path.exists('importances.csv'):
        RF(x, y, pre_x)
    df = pd.read_csv('importances.csv')
    importances = df.imp
    indices = np.argsort(importances)[::-1]
    n = int(len(indices) * rate)
    x = x[:, indices[0:n]]
    pre_x = pre_x[:, indices[0:n]]
    return x,  pre_x


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


def XGB(x, y, pre_x):
    print '----xgb-----'
    x, pre_x = feature_select(x, y, pre_x, rate=0.3)
    print x.shape
    print pre_x.shape

    posnum = y[y == 1].shape[0]
    negnum = y[y == 0].shape[0]
    print 'pos:', posnum, ' neg:', negnum

    weight = float(posnum) / (posnum + negnum)
    print 'weight:', weight

    xtrain, xvalid, ytrain, yvalid = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y)

    if not submit_flag:
        xtrain, xtest, ytrain, ytest = train_test_split(
            xtrain, ytrain, test_size=0.2, random_state=0)
        dtest = xgb.DMatrix(xtest, label=ytest, missing=-1)

    dtrain = xgb.DMatrix(xtrain, label=ytrain, missing=-1)
    dvalid = xgb.DMatrix(xvalid, label=yvalid, missing=-1)

    dpre = xgb.DMatrix(pre_x)

    param = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'early_stopping_rounds': 100,
        'eval_metric': 'logloss',
        'max_depth': 6,
        'silent': 1,
        'eta': 0.05,
        'nthread': 16,
        'scale_pos_weight': weight
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'val')]
    model = xgb.train(param, dtrain, num_boost_round=500, evals=watchlist)

    # valid
    valid_pre = model.predict(dvalid, ntree_limit=model.best_iteration)
    logloss(yvalid, valid_pre)

    if not submit_flag:
        test_pre = model.predict(dtest, ntree_limit=model.best_iteration)
        logloss(ytest, test_pre)

    # predict
    pre_y = model.predict(dpre, ntree_limit=model.best_iteration)

    return pre_y


def NN(X, y):
    pass

def save_pred(ypre, inst):
    df = pd.concat([ypre, inst], axis=0)
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    x, y, xpre, inst = load_feature(from_file=True, with_ohe=False)
    # xgboost
    ypre = XGB(x, y, xpre)
    save_pred(ypre, inst)
    # random forest
    # ypre = rfpredict(x, y, xpre)