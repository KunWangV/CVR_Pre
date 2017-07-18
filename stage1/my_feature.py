# coding: utf-8
import pandas as pd
import numpy as np
from data_utils import *
from feature import *

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from model_direct import *
from sklearn import preprocessing


def min_max(xtrain, xtest):
    """
    :param xtrain: numpy
    :param xtest: numpy
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(np.concatenate(xtrain.reshape(-1, 1), xtest.reshape(-1, 1)))
    x_train = min_max_scaler.transform(xtrain)
    x_test = min_max_scaler.transform(xtest)
    return x_train, x_test


def comb(train_a, train_b, _train_y, test_a, test_b, feat_a, feat_b):
    """
    组合特征
    :param full:
    :param train_a
    :param train_b, , test_b, feat_a, feat_b
    :param _train_y:
    :param test_a:
    :param test_b:
    :param feat_a:
    :param feat_b:


    :return:
    """
    if not (feat_a in real_feats and feat_b in real_feats):
        if feat_a in real_cvt_feats:
            train_a = pd.Series(np.floor(train_a / 0.05))
            test_a = pd.Series(np.floor(test_a / 0.05))
        if feat_b in real_cvt_feats:
            train_b = pd.Series(np.floor(train_b / 0.05))
            test_b = pd.Series(np.floor(test_b / 0.05))
        _x_train = np.asarray(
            (train_a.astype(str) + train_b.astype(str)).astype('category').values.codes)
        _x_test = np.asarray(
            (test_a.astype(str) + test_b.astype(str)).astype('category').values.codes)
        if feat_a + '|' + feat_b in cate_low_dim:
            return ohe(x_train, train_y, x_test)
        else:
            return embed(
                _x_train.reshape(-1, 1),
                _train_y.values.reshape(-1, 1), _x_test.reshape(-1, 1))

    else:
        train_a, test_a = min_max(train_a.values, test_a.values)
        train_b, test_b = min_max(train_b.values, test_b.values)
        return train_a * train_b, _train_y, test_a * test_b


def ohe(X_train, y_train, X_test):
    """
    独热
    :param full:
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    enc = OneHotEncoder()
    enc.fit(np.concatenate((X_train, X_test), axis=0).values.reshape(-1, 1))
    ohe_x_train = enc.transform(X_train)
    ohe_x_test = enc.transform(X_test)
    return ohe_x_train, y_train, ohe_x_test


def embed(X_train, y_train, X_test):
    """
    拉成向量
    :param full:
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    n_estimator = 10
    enc = OneHotEncoder()
    enc.fit(np.concatenate((X_train, X_test), axis=0).reshape(-1, 1))
    ohe_x_train = enc.transform(X_train)
    ohe_x_test = enc.transform(X_test)
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd.fit(ohe_x_train, y_train.ravel())

    grd_enc = OneHotEncoder()
    grd_enc.fit(grd.apply(ohe_x_train)[:, :, 0])

    rst_x_train = grd_enc.transform(grd.apply(ohe_x_train)[:, :, 0])
    rst_x_teset = grd_enc.transform(grd.apply(ohe_x_test)[:, :, 0])

    return rst_x_train, y_train.ravel(), rst_x_teset


chosen_features = [
    'cvt_positionID',
    'pre_cvt_userID',
    # 'cnt_creativeID_positionID',
    'cvt_creativeID',
    'cvt_userID',
    'postion_cri_day_click_cnt',
    'clickTime_day',
    'pre_cvt_camgaignID',
    'pre_cvt_creativeID',
    'is_day_rpt_last_click',
    'is_day_rpt_first_click',
    'cnt_userID',
    'cvt_camgaignID',
    'cvt_connectionType',
    'appID',
    'position_day_click_cnt',
    'rpt_day_click_cnt',
    'inst_app_installed',
    'cnt_connectionType',
    'pre_cvt_positionID',
    'action_cate',
    'cvt_advertiserID',
    'pre_cnt_positionID',
    'pre_cnt_userID',
    'appCategory',
    'tt_cnt_appcate',
    'cvt_sitesetID',
    'pre_cvt_connectionType',
    'cnt_positionID',
    'rpt_click_cnt',
    'cnt_appID',
    'cnt_advertiserID',
    'user_day_click_cnt',
    'cvt_appCategory',
    'age',
    'gender',
    'pre_cnt_advertiserID',
    'advertiserID',
    'positionID',
    'postion_app_day_click_cnt',
    'pre_cvt_advertiserID',
    'pre_cnt_creativeID',
    'cnt_appCategory',
    'cvt_adID',
    'cvt_appID',
    'connectionType',
    'is_rpt_first_click',
    'cnt_appPlatform',
    'pre_cnt_appID',
    'cnt_adID',
    'cri_day_click_cnt',
    'cnt_creativeID',
    'pre_cnt_appCategory',
]

if not os.path.exists("X_train.npz"):
    real_cvt_feats = [
        'cvt_userID',
        'pre_cvt_userID',
        'cvt_creativeID',
        'pre_cvt_creativeID',
        'cvt_positionID',
        'pre_cvt_positionID',
        'cvt_adID',
        'pre_cvt_adID',
        'cvt_camgaignID',
        'pre_cvt_camgaignID',
        'cvt_advertiserID',
        'pre_cvt_advertiserID',
        'cvt_appID',
        'pre_cvt_appID',
        'cvt_sitesetID',
        'pre_cvt_sitesetID',
        'cvt_appCategory',
        'pre_cvt_appCategory',
        'cvt_appPlatform',
        'pre_cvt_appPlatform',
        'cvt_education',
        'pre_cvt_education',
        'cvt_gender',
        'pre_cvt_gender',
        'cvt_haveBaby',
        'pre_cvt_haveBaby',
        'cvt_marriageStatus',
        'pre_cvt_marriageStatus',
        'cvt_positionType',
        'pre_cvt_positionType',
        'cvt_hometown_c',
        'pre_cvt_hometown_c',
        'cvt_hometown_p',
        'pre_cvt_hometown_p',
        'cvt_residence_c',
        'pre_cvt_residence_c',
        'cvt_residence_p',
        'pre_cvt_residence_p',
        'cvt_telecomsOperator',
        'pre_cvt_telecomsOperator',
        'cvt_connectionType',
        'pre_cvt_connectionType',
        'cvt_clickTime_week',
        # 'pre_cvt_clickTime_week',
    ]

    real_cnt_feats = [
        'app_day_click_cnt',
        'app_uuser_click_cnt',
        'cri_day_click_cnt',
        'cri_uuser_click_cnt',
        'inst_cnt_appcate',
        'inst_cnt_installed',
        'position_day_click_cnt',
        'postion_app_day_click_cnt',
        'postion_cri_day_click_cnt',
        'rpt_click_cnt',
        'rpt_day_click_cnt',
        'tt_cnt_appcate',
        'user_app_day_click_cnt',
        'user_cri_day_click_cnt',
        'user_day_click_cnt',
        'cnt_userID',
        'pre_cnt_userID',
        'cnt_creativeID',
        'pre_cnt_creativeID',
        'cnt_positionID',
        'pre_cnt_positionID',
        'cnt_adID',
        'pre_cnt_adID',
        'cnt_camgaignID',
        'pre_cnt_camgaignID',
        'cnt_advertiserID',
        'pre_cnt_advertiserID',
        'cnt_appID',
        'pre_cnt_appID',
        'cnt_sitesetID',
        'pre_cnt_sitesetID',
        'cnt_appCategory',
        'pre_cnt_appCategory',
        'cnt_appPlatform',
        'pre_cnt_appPlatform',
        'cnt_education',
        'pre_cnt_education',
        'cnt_gender',
        'pre_cnt_gender',
        'cnt_haveBaby',
        'pre_cnt_haveBaby',
        'cnt_marriageStatus',
        'pre_cnt_marriageStatus',
        'cnt_positionType',
        'pre_cnt_positionType',
        'cnt_hometown_c',
        'pre_cnt_hometown_c',
        'cnt_hometown_p',
        'pre_cnt_hometown_p',
        'cnt_residence_c',
        'pre_cnt_residence_c',
        'cnt_residence_p',
        'pre_cnt_residence_p',
        'cnt_telecomsOperator',
        'pre_cnt_telecomsOperator',
        'cnt_connectionType',
        'pre_cnt_connectionType',
        'cnt_clickTime_week',
        'pre_cnt_clickTime_week',
        # 'cnt_creativeID_positionID',
    ]

    real_other = [
        'inst_app_installed',
        'inst_cate_percent',
        'action_cate',
        'action_cate_recent',
        'action_installed',
    ]

    cate_low_dim = [
        'age',
        'appCategory',
        'appPlatform',
        'clickTime_day',
        'clickTime_hour',
        'clickTime_minute',
        'clickTime_week',
        'connectionType',
        'education',
        'gender',
        'haveBaby',
        'hometown_c',
        'hometown_p',
        'marriageStatus',
        'positionType',
        'telecomsOperator',
        'residence_c',
        'residence_p',
        'appID',
        'inst_is_installed',
        'is_day_rpt_first_click',
        'is_day_rpt_last_click',
        'is_rpt_first_click',
        'is_rpt_last_click',
        'tt_is_installed',
        'sitesetID',
    ]

    cate_high_dim = [
        'adID',
        'advertiserID',
        'camgaignID',
        'creativeID',
        'positionID',
        # 'userID',
    ]

    cate_feats = cate_high_dim + cate_low_dim
    real_feats = real_cnt_feats + real_cvt_feats + real_other
    drop_feats = [
        'cnt_creativeID_positionID',
    ]

    feats = cate_feats + real_feats

    train = read_as_pandas('train.csv')
    train = train.loc[(train['clickTime_day'] >= 17) & (train['clickTime_day']
                                                        <= 30), :]
    test = read_as_pandas('test.csv')
    print train.shape, test.shape

    df_concate = pd.concat([train, test], axis=0).reset_index(drop=True)
    df_concate['age'] = df_concate['age'] // 5
    transform_category(df_concate, cate_feats)
    df_concate = df_concate.fillna(0)
    # transform_fillna(df_concate, cate_feats, value=0)
    # transform_fillna(df_concate, real_cnt_feats, value=0)

    train = df_concate.loc[df_concate['instanceID'] == 0, :].drop(
        'instanceID', axis=1).reset_index(drop=True)
    test = df_concate.loc[df_concate['instanceID'] > 0, :].sort_values(
        'instanceID').reset_index(drop=True)

    print train.shape, test.shape

    days = df_concate.clickTime_day.unique()
    print days
    train_data = train['clickTime_day'] <= days[-4]
    print days[-3]
    print train['clickTime_day'].unique()
    eval_data = np.logical_or(train['clickTime_day'] == days[-3],
                              train['clickTime_day'] == days[-2])
    # eval_data = train['clickTime_day'] == days[-2]
    train_idx = train.loc[train_data].index.tolist()
    eval_idx = train.loc[eval_data].index.tolist()

    id = test['instanceID'].values
    print id
    y_train = train['label'].values[train_idx]
    print y_train
    y_eval = train['label'].values[eval_idx]
    enc = OneHotEncoder()

    print 'features used: '
    flag = True
    for i, feat in enumerate(cate_low_dim):
        if feat in drop_feats:
            continue
        if feat not in chosen_features:
            continue
        print feat
        enc.fit(
            np.concatenate((train[feat].values.reshape(-1, 1), test[feat]
                            .values.reshape(-1, 1))))
        x_train = enc.transform(train[feat].values.reshape(-1, 1))
        x_test = enc.transform(test[feat].values.reshape(-1, 1))
        # print x_train.shape, x_test.shape
        if flag:
            X_train, X_eval, X_test = x_train[train_idx], x_train[
                eval_idx], x_test
            flag = False
        else:
            X_train, X_eval, X_test = sparse.hstack(
                (X_train, x_train[train_idx])), sparse.hstack(
                (X_eval, x_train[eval_idx])), sparse.hstack((X_test,
                                                             x_test))

        print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape

    print 'real features'
    start = X_train.shape[1]
    for i, feat in enumerate(real_feats):
        if feat in drop_feats:
            continue
        if feat not in chosen_features:
            continue
        print feat
        x_train = train[feat].values.reshape(-1, 1)
        x_test = test[feat].values.reshape(-1, 1)
        if x_train.dtype == np.float64:
            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(
            np.concatenate((train[feat].values.reshape(-1, 1), test[feat]
                            .values.reshape(-1, 1))))
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)

        X_train, X_eval, X_test = sparse.hstack(
            (X_train, x_train[train_idx])), sparse.hstack(
            (X_eval, x_train[eval_idx])), sparse.hstack((X_test, x_test))

        print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape

    # 做多项式
    end = X_train.shape[1]
    poly = PolynomialFeatures(2, interaction_only=True)
    _np_x_train = np.asarray(X_train.todense())
    _np_x_test = np.asarray(X_test.todense())
    poly.fit(_np_x_train[:, start:end-10])
    x_train = poly.transform(_np_x_train[:, start:end-10])
    x_test = poly.transform(_np_x_test[:, start:end-10])
    del _np_x_train
    del _np_x_test
    X_train, X_eval, X_test = sparse.hstack(
        (X_train, x_train[train_idx])), sparse.hstack(
        (X_eval, x_train[eval_idx])), sparse.hstack((X_test, x_test))

    for i, feat in enumerate(cate_high_dim):
        if feat in drop_feats:
            continue
        if feat not in chosen_features:
            continue
        print feat + '_emb'
        x_train, _, x_test = embed(
            train[feat].values.reshape(-1, 1), train['label'].values.reshape(
                -1, 1), test[feat].values.reshape(-1, 1))
        X_train, X_eval, X_test = sparse.hstack(
            (X_train, x_train[train_idx])), sparse.hstack(
            (X_eval, x_train[eval_idx])), sparse.hstack((X_test, x_test))

        print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape

    # combine_low_dim = [
    #     # 'advertiserID|connectionType',  # 作为string 连接  再OHE
    #     # 'pre_cvt_camgaignID|pre_cvt_creativeID',  # (分桶 0.05)作为string 连接 OHE
    #     # 'cvt_userID|user_day_click_cnt',  # cvt_userID|(分桶 0.05)作为string 和user_day_click_cnt 作为string连接 再OHE
    #     # 'cvt_userID|is_rpt_last_click',  # cvt_userID|(分桶 0.05)作为string 和is_rpt_last_click  作为string连接 再OHE
    #     # 'pre_cvt_camgaignID|pre_cvt_userID',  # (分桶 0.05)作为string 连接 OHE
    #     # 'is_rpt_first_click|user_day_click_cnt',  # 作为string 连接  再OHE
    #     'pre_cvt_creativeID|pre_cvt_userID',  # (分桶 0.05)作为string 连接 OHE
    #     'pre_cvt_userID|user_day_click_cnt',  # pre_cvt_userID(分桶 0.05)作为string 和user_day_click_cnt    作为
    # ]
    # combine_high_dim = [
    #     'creativeID|positionID',  # 作为string 连接 再embedding
    #     'connectionType|positionID',  # 作为string 连接 再embedding
    #     'creativeID|pre_cvt_userID',  # creativeID作为string 和  pre_cvt_userID(分桶 0.05)作为string 连接  再embedding
    #     'positionID|pre_cvt_userID',  # positionID作为string 和  pre_cvt_userID(分桶 0.05)作为string 连接  再embedding
    #     'positionID|pre_cvt_creativeID',  # positionID作为string 和  pre_cvt_creativeID(分桶 0.05)作为string 连接  再embedding
    #     'positionID|pre_cvt_camgaignID',  # positionID作为string 和  pre_cvt_camgaignID(分桶 0.05)作为string 连接  再embedding
    #     'camgaignID|positionID',  # 作为string 连接 再embedding
    #     'camgaignID|creativeID',  # 作为string 连接 再embedding
    #     # 'creativeID|pre_cvt_camgaignID',  # creativeID|作为string 和  pre_cvt_camgaignID(分桶 0.05)作为string 连接  再embedding
    # ]
    #
    # for i, feat in enumerate(combine_low_dim):
    #     print feat + '_combine'
    #     feat_a, feat_b = feat.split('|')
    #     x_train, _, x_test = comb(train[feat_a], train[feat_b],
    #                               train['label'], test[feat_a],
    #                               test[feat_b], feat_a, feat_b)
    #     X_train, X_eval, X_test = sparse.hstack(
    #         (X_train, x_train[train_idx])), sparse.hstack(
    #         (X_eval, x_train[eval_idx])), sparse.hstack((X_test, x_test))
    #
    # print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape
    #
    # for i, feat in enumerate(combine_high_dim):
    #     print feat + '_combine'
    #     feat_a, feat_b = feat.split('|')
    #     x_train, _, x_test = comb(train[feat_a], train[feat_b],
    #                               train['label'], test[feat_a],
    #                               test[feat_b], feat_a, feat_b)
    #     X_train, X_eval, X_test = sparse.hstack(
    #         (X_train, x_train[train_idx])), sparse.hstack(
    #         (X_eval, x_train[eval_idx])), sparse.hstack((X_test, x_test))
    #
    # print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape

    sparse.save_npz('X_train.npz', X_train)
    sparse.save_npz('X_eval.npz', X_eval)
    sparse.save_npz('y_train.npz', sparse.csr_matrix(y_train))
    sparse.save_npz('y_eval.npz', sparse.csr_matrix(y_eval))
    sparse.save_npz('X_test.npz', X_test)

else:
    X_train = sparse.load_npz('X_train.npz')
    X_eval = sparse.load_npz('X_eval.npz')
    y_train = sparse.load_npz('y_train.npz')
    y_eval = sparse.load_npz('y_eval.npz')
    X_test = sparse.load_npz('X_test.npz')

weight1 = 93262 / 3749528.0
print 'weight:', weight1
print X_train.shape, X_eval.shape, y_train.shape, y_eval.shape, X_test.shape
# X_train = np.asarray(X_train)
# X_eval = np.asarray(X_eval)
y_train = np.asarray(y_train)
y_eval = np.asarray(y_eval)
# X_test = np.asarray(X_test)

y_pre, _ = LGB(
    X_train.tocsr(), X_eval.tocsr(), y_train, y_eval, X_test.tocsr(), use_gpu=False, weight=weight1)
id = id.astype(int)
save_pred(y_pre, inst=id, subffix='lgb')

# y_pre, _ = LR(X_train, X_eval, y_train, y_eval, X_test)
# save_pred(y_pre, inst=id, subffix='lr')

# y_pre, _ = XGB(
#     X_train, X_eval, y_train, y_eval, X_test, use_gpu=True, weight=weight1)
# save_pred(y_pre, inst=id, subffix='xgb')

# dtrain = xgboost.DMatrix(X_train, label=y_train)
# deval = xgboost.DMatrix(X_eval, label=y_eval)
# dtest = xgboost.DMatrix(X_test)
#
# param = {
#     'booster': 'gbtree',
#     'objective': 'binary:logistic',
#     'gamma': 0.1,
#     'max_depth': 6,
#     'lambda': 2,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'min_child_weight': 5,
#     'silent': 0,
#     # 'eta': 0.01,
#     'seed': 1000,
#     'nthread': 7,
#     'eval_metric': 'logloss',
#     'scale_pos_weight': weight,
#     'early_stopping_rounds': 10
# }
#
# # specify validations set to watch performance
# watchlist = [(deval, 'eval'), (dtrain, 'train')]
# num_round = 200
# bst = xgboost.train(param, dtrain, num_round, watchlist)
#
# proba_eval = np.array(bst.predict(deval))
# print logloss(y_eval, np.array(proba_eval))
#
# proba_test = np.array(bst.predict(dtest))
# save_pred(proba_test, id)
