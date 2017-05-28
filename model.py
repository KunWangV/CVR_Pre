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
from Xgboost_Feature import XgboostFeature

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from data import *
from feature import *
import os

submit_flag = True


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(
        sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    print '-------------logloss-----------', ll
    return ll


def feature_select(x_train, x_test, y_train, y_test, x_pre, rate=0.2):
    # RF(x_train, y_train, pre_x)
    if not os.path.exists('importances.csv'):
        RF(x_train, y_train, x_pre)
    df = pd.read_csv('importances.csv')
    importances = df.imp
    indices = np.argsort(importances)[::-1]
    print 'select indices: '
    print indices
    n = int(len(indices) * rate)
    print 'n'
    print n
    print 'x_train.shape'
    print x_train.shape
    print 'x_test.shape'
    print x_test.shape
    print 'x_pre.shape'
    print x_pre.shape
    x_train = x_train[:, indices[0:n]]
    x_test = x_test[:, indices[0:n]]
    x_pre = x_pre[:, indices[0:n]]
    return x_train, x_test, y_train, y_test, x_pre


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


def NewFeatrue(x_train, x_test, y_train, y_test, x_pre):
    # 自己设置xgboost模型参数 默认树个数30
    model = XgboostFeature(n_estimators=200, learning_rate=0.1, max_depth=7, min_child_weight=5, gamma=0.3, subsample=0.9,
                           colsample_bytree=0.9, objective='binary:logistic', nthread=4, scale_pos_weight=1, reg_alpha=1e-05, reg_lambda=1, seed=27)
    # 切分训练集训练叶子特征模型 返回值是 原特征+新特征
    x_train, y_train, x_test, y_test, x_pre = model.fit_model_split(
        x_train, y_train, x_test, y_test, x_pre)
    # 不切分训练集训练叶子特征模型  返回值 是原特征+新特征
    # X_train,y_train, X_test, y_test=model.fit_model(X_train, y_train,X_test, y_test)
    return x_train, x_test, y_train, y_test, x_pre


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

    print x.shape
    print y.shape
    xtrain, xvalid, ytrain, yvalid = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y)

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


def XGB(xtrain, xvalid, ytrain, yvalid, pre_x):
    print '----xgb-----'

    if not submit_flag:
        xtrain, xtest, ytrain, ytest = train_test_split(
            xtrain, ytrain, test_size=0.2, random_state=0, stratify=y)
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


def save_pred(ypre, inst):
    df = pd.DataFrame({'instanceID': inst, 'prob': ypre})
    print 'df:'
    print df.shape
    df.to_csv('submission.csv', index=False)


def LR(xtrain, xvalid, ytrain, yvalid, pre_x):
    print '----LR-----'

    if not submit_flag:
        xtrain, xtest, ytrain, ytest = train_test_split(
            xtrain, ytrain, test_size=0.2, random_state=0, stratify=y)
        dtest = xgb.DMatrix(xtest, label=ytest, missing=-1)

    sc = StandardScaler()
    sc.fit(xtrain)  # 估算每个特征的平均值和标准差
    X_train_std = sc.transform(xtrain)
    # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    X_valid_std = sc.transform(xvalid)

    model = LogisticRegression(C=50.0, random_state=0)
    model.fit(X_train_std, ytrain)

    # valid
    valid_pre = model.predict_proba(X_valid_std)
    logloss(yvalid, valid_pre[:, 1])

    if not submit_flag:
        test_pre = model.predict(dtest)
        logloss(ytest, test_pre)

    # predict
    pre_y = model.predict(pre_x)

    return pre_y


def LGB(xtrain, xvalid, ytrain, yvalid, pre_x):
    if not submit_flag:
        print "lgb: split train and test"
        xtrain, xtest, ytrain, ytest = train_test_split(
            xtrain, ytrain, test_size=0.2, random_state=0, stratify=y)

    posnum = ytrain[ytrain == 1].shape[0]
    negnum = ytrain[ytrain == 0].shape[0]
    print 'pos:', posnum, ' neg:', negnum

    weight = float(posnum) / (posnum + negnum)
    print 'weight:', weight

    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_eval = lgb.Dataset(xvalid, yvalid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'feature_fraction': 0.5,
        'is_unbalance': True,
        'scale_pos_weight': weight,
        'max_depth': 6,
        'verbose': -1,
        'num_threads': 16
    }

    #feature_name = ['feature_' + str(col) for col in range(num_feature)]

    print '----start training----'

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100)

    # gbm.save_model('gbm_model.txt')

    #bst = lgb.Booster(model_file='gbm_model.txt')

    valid_pre = gbm.predict(xvalid, num_iteration=gbm.best_iteration)
    print '-----valid-----'
    print 'not threshold: '
    logloss(yvalid, valid_pre)
    print 'threshold: '
    valid_pre = threshold(valid_pre)
    logloss(yvalid, valid_pre)
    y_pre = gbm.predict(pre_x, num_iteration=gbm.best_iteration)
    print ' not thresholding final submission'
    # y_pre = threshold(y_pre)
    return y_pre


if __name__ == '__main__':

    # 导入数据
    train_x, train_y, test_x, test_y, pre_x, inst_id = load_feature(from_file=False, with_ohe=False,modelType='LGBM',test_days=2)
    # 数据分割为测试集和训练集
    #x_train, x_test, y_train, y_test = split_train_test_by_day(x, y, test_day_size=2)  # test_size测试集合所占比例
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

    # xgboost
    # ypre = XGB(x_train, x_test, y_train, y_test, x_pre)
    # LR
    # ypre = LR(x_train, x_test, y_train, y_test, x_pre)
    # LGB
    ypre = LGB(train_x, test_x, train_y, test_y, pre_x)

    # 保存结果
    inst_id=inst_id.astype('int64')
    save_pred(ypre, inst_id)
    # ypre = XGB(x, y, xpre)

    # random forest
    # ypre = rfpredict(x, y, xpre)
