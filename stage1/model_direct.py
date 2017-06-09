# coding: utf-8
# pylint: disable=C0103, C0111,C0326
import scipy as sp
# import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import lightgbm as lgb

from sklearn import metrics
from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeavePGroupsOut

from data import *
from feature import *
from to_FFM import *
from to_LR import *
from to_LGB_direct import *
import os

submit_flag = True


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) +
             sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    print '-------------logloss-----------', ll
    return ll


def cross_validation(X, y, pre_x, groups, model='LGB', test_days=1):
    groups = np.floor((groups + 1) / 2)

    logo = LeavePGroupsOut(n_groups=test_days)
    i = 0
    pre_sum = np.zeros(pre_x.shape[0])
    pre_ = []
    print np.isnan(groups).astype(int).sum()
    print np.unique(groups)
    ll_ = []
    for train, test in logo.split(X, y, groups=groups):
        i = i + 1
        print 'times:', i
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        print X_train.shape, X_test.shape, y_train.shape, y_test.shape
        if model == 'LGB':
            pre, ll = LGB(X_train, X_test, y_train, y_test, pre_x)
        else:
            pre, ll = LR(X_train, X_test, y_train, y_test, pre_x)
        ll_ += [ll]
        pre_ += [pre]
    weight = []
    weight_sum = 0
    for l in ll_:
        weight_sum += 1.0 / l
        weight += [1.0 / l]
    for i in range(len(pre_)):
        pre_sum += pre_[i] * weight[i] / weight_sum

    print 'weight', weight
    print 'loss', ll_

    return pre_sum


def threshold(y, thresh=0.005, val=1e-6):
    y[y <= thresh] = val
    return y


def submit(y, inst):
    y[y < 0] = 0
    now = time.strftime('%Y%m%d%H%M%S')
    data = pd.DataFrame(inst, columns=['instanceID'])
    data['prob'] = y
    data.instanceID = np.round(data.instanceID).astype(int)
    data = data.sort(['instanceID'], ascending=True)
    data.to_csv('../res/' + now + '.csv', index=False)


def RF(train_x, train_y, test_x, test_y, pre_x):
    """
    随机森林，计算特征重要性
    """
    print '----rf-----'
    # print x.shape
    # print pred_x.shape

    # posnum = y[y == 1].shape[0]
    # negnum = y[y == 0].shape[0]
    # print 'pos:', posnum, ' neg:', negnum
    # weight = float(posnum) / (posnum + negnum)
    # print 'weight:', weight

    # print x.shape
    # print y.shape
    # xtrain, xvalid, ytrain, yvalid = train_test_split(
    #     x, y, test_size=0.2, random_state=0, stratify=y)

    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=6,
                                 random_state=500,
                                 class_weight={1: weight},
                                 n_jobs=8) \
        .fit(train_x, train_y)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print indices
    print importances[indices]

    imp = pd.DataFrame(importances, columns=['imp'])
    imp.to_csv('importances.csv', index=False)

    test_pred = clf.predict_proba(test_x)
    logloss(test_y, test_pred[:, 1])
    pred = clf.predict_proba(pre_x)
    return pred[:, 1], indices


def XGB(xtrain, xvalid, ytrain, yvalid, pre_x, weight=1, use_gpu=False):
    print '----xgb-----'
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

    if use_gpu:
        param['gpu_id'] = 0
        param['max_bin'] = 16
        param['updater'] = 'grow_gpu'

    watchlist = [(dtrain, 'train'), (dvalid, 'val')]
    model = xgb.train(param, dtrain, num_boost_round=200, evals=watchlist)

    # valid
    print model.best_iteration
    valid_pre = model.predict(dvalid, ntree_limit=model.best_iteration)
    logloss(yvalid, valid_pre)

    if not submit_flag:
        test_pre = model.predict(dtest, ntree_limit=model.best_iteration)
        logloss(ytest, test_pre)

    # predict
    pre_y = model.predict(dpre, ntree_limit=model.best_iteration)

    return pre_y


def deep_and_wide(X, y):
    pass


def save_pred(ypre, inst, subffix=''):
    print len(inst)
    print len(ypre)
    df = pd.DataFrame({'instanceID': inst, 'prob': ypre})
    print 'df:'
    print df.shape
    df.to_csv('submission.{}.csv'.format(subffix), index=False)


def LR(xtrain, xvalid, ytrain, yvalid, pre_x):
    print '----LR-----'

    # sc = StandardScaler()
    # sc.fit(xtrain)  # 估算每个特征的平均值和标准差
    # X_train_std = sc.transform(xtrain)
    # # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    # X_valid_std = sc.transform(xvalid)

    model = LogisticRegression(C=50.0, random_state=0)
    model.fit(xtrain, ytrain)

    # valid
    valid_pre = model.predict_proba(xvalid)
    ll = logloss(yvalid, valid_pre[:, 1])

    # predict
    pre_y = model.predict_proba(pre_x)[:, 1]

    return pre_y, ll
 

def LGB(xtrain, xvalid, xtest, ytrain, yvalid, ytest, pre_x, weight=1, use_gpu=False):
    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_eval = lgb.Dataset(xvalid, yvalid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'feature_fraction': 0.5,
        'is_unbalance': True,
        'scale_pos_weight': weight,
        'max_depth': 6,
        'verbose': -1,
        # 'num_threads': 8,
    }
    if use_gpu:
        params['device'] = 'gpu'
        # params['max_bin'] =

    # feature_name = ['feature_' + str(col) for col in range(num_feature)]

    print '----start training----'

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_eval, lgb_train],
        valid_names=['eval', 'train'],
        early_stopping_rounds=100)

    # gbm.save_model('gbm_model.txt')

    # bst = lgb.Booster(model_file='gbm_model.txt')

    valid_pre = gbm.predict(xvalid, num_iteration=gbm.best_iteration)
    # print '-----valid-----'
    # print 'not threshold: '
    print '29/30'
    ll = logloss(yvalid, valid_pre)
    # print 'threshold: '
    # valid_pre = threshold(valid_pre)
    # logloss(yvalid, valid_pre)
    test_pre = gbm.predict(xtest, num_iteration=gbm.best_iteration)
    df = pd.DataFrame({'prob': test_pre})
    df.to_csv('_29_pro.csv')
    print '29'
    ll1 = logloss(ytest, test_pre)

    y_pre = gbm.predict(pre_x, num_iteration=gbm.best_iteration)
    # print ' not thresholding final submission'
    # y_pre = threshold(y_pre)
    return y_pre, ll, ll1


def LGB(xtrain, xvalid, ytrain, yvalid, pre_x, use_gpu=False, weight=0.1):
    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_eval = lgb.Dataset(xvalid, yvalid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'feature_fraction': 0.5,
        'is_unbalance': True,
        'scale_pos_weight': weight,
        'max_depth': 6,
        'verbose': -1,
        'num_threads': 16,
    }
    if use_gpu:
        params['device'] = 'gpu'
        # params['max_bin'] =

    # feature_name = ['feature_' + str(col) for col in range(num_feature)]

    print '----start training----'

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_eval, lgb_train],
        valid_names=['eval', 'train'],
        early_stopping_rounds=100)

    # gbm.save_model('gbm_model.txt')

    # bst = lgb.Booster(model_file='gbm_model.txt')

    valid_pre = gbm.predict(xvalid, num_iteration=gbm.best_iteration)
    # print '-----valid-----'
    # print 'not threshold: '
    print '29/30'
    ll = logloss(yvalid, valid_pre)
    df = pd.DataFrame({'prob': valid_pre})
    df.to_csv('valid_pre.csv')
    # print 'threshold: '
    # valid_pre = threshold(valid_pre)
    # logloss(yvalid, valid_pre)

    y_pre = gbm.predict(pre_x, num_iteration=gbm.best_iteration)
    # print ' not thresholding final submission'
    # y_pre = threshold(y_pre)
    return y_pre, ll


if __name__ == '__main__':
    if_CV = False
    # 导入数据
    df_train, df_pre = load_feature(from_file=False, with_ohe=False)

    # 数据分割为测试集和训练集
    # x_train, x_test, y_train, y_test = split_train_test_by_day(x, y, test_day_size=2)  # test_size测试集合所占比例
    # 使用XGBoost构造新特征
    # x_train, x_test, y_train, y_test, x_pre = NewFeatrue(
    #     x_train, x_test, y_train, y_test, x_pre)
    # print 'x_train.shape'
    # print x_train.shape
    # print 'x_test.shape'
    # print x_test.shape
    # print 'x_pre.shape'
    # print x_pre.shape
    # 使用RF选择重要特征
    # x_train, x_test, y_train, y_test, x_pre = feature_select(
    #     x_train, x_test, y_train, y_test, x_pre, rate=0.8)

    # posnum = train_y[train_y == 1].shape[0]
    # negnum = train_y[train_y == 0].shape[0]
    # print 'pos:', posnum, ' neg:', negnum

    # weight = float(posnum) / (posnum + negnum)
    # print 'weight:', weight

    weight = 93262 / 3749528.0
    print 'weight:', weight

    # xgboost
    # ypre = XGB(x_train, x_test, y_train, y_test, x_pre)
    # ypre = XGB(train_x, test_x, train_y, test_y, pre_x)
    # LR
    # if if_CV:
    #     x, y, groups, pre_x, inst_id = to_LR(df_train, df_pre, with_ohe=True, if_CV=False)
    #     ypre = cross_validation(x, y, pre_x, groups, model='LR', test_days=1)
    # else:
    #     train_x, train_y, test_x, test_y, pre_x, inst_id = to_LR(df_train, df_pre, with_ohe=True, test_days=2,
    #                                                              if_CV=False)
    #     ypre, _ = LR(train_x, test_x, train_y, test_y, pre_x)
    # FFM
    # to_FFM(df_train, df_pre, with_ohe=False, if_CV=True)
    # LGB
    if if_CV:
        x, y, groups, pre_x, inst_id = to_LGBM(df_train, df_pre, if_CV=True)
        ypre = cross_validation(x, y, pre_x, groups, model='LGB', test_days=1)
    else:
        train_x, train_y, test_x, test_y, test_x1, test_y1, pre_x, inst_id = to_LGBM(
            df_train, df_pre, test_days=2, if_CV=False)
        ypre, _, _ = LGB(train_x, test_x, test_x1, train_y, test_y, test_y1,
                         pre_x)

        # RF(train_x, train_y, test_x, test_y, pre_x)

        # x, y, groups, pre_x, inst_id = to_LGBM(df_train, df_pre, test_days=2, if_CV=True)
        # split_train_test(x, y, test_size=0.2, stratify=True, with_df=False)
        # train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.1,stratify=y)

    # 保存结果
    inst_id = inst_id.astype(int)
    save_pred(ypre, inst_id)
    # ypre = XGB(x, y, xpre)

    # random forest
    # ypre = rfpredict(x, y, xpre)
