# coding: utf-8
# pylint: disable=C0103, C0111

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

import pandas as pd
#from pyspark.ml.feature import OneHotEncoder as OHE
#import pyspark.sql.functions as F
import numpy as np

from data_utils import *
import pdb
import pickle

#from tqdm import tqdm
import os

def to_LR(df_train, df_pre, with_ohe=True, test_days=2, if_CV=True):
    """
        cvt_positionID (5454,) 0.0 1.0
        cvt_userID (51,) 0.0 1.0
        cvt_creativeID (3880,) 0.0 1.0
        cvt_camgaignID (2377,) 0.0 1.0
        clickTime_day (14,) 17 30
        cvt_connectionType (65,) 0.00402765444628 0.0317665577059
        cnt_userID (87,) 0.0 111.0
        cvt_advertiserID (574,) 0.0 0.702702702703
        cnt_connectionType (65,) 2894.0 2726052.0
        cvt_sitesetID (39,) 0.0161008932448 0.0476605066562
        action_cate (88,) 0 117
        tt_cnt_appcate (121,) 0 250
        cnt_positionID (3059,) 0.0 351373.0
        appID (50,) 14 472
        cnt_appID (443,) 0.0 1765710.0
        appCategory (14,) 0 503
        cnt_advertiserID (625,) 0.0 1765710.0
        cvt_adID (3267,) 0.0 1.0
        age (81,) 0 80
        inst_app_installed (13,) 0 282777
        gender (3,) 0 2
        cvt_gender (39,) 0.0191151849487 0.0309996027044
        cnt_camgaignID (2109,) 0.0 554534.0
        cnt_appCategory (160,) 10.0 1765710.0
        advertiserID (89,) 1 91
        positionType (6,) 0 5
        cvt_appID (399,) 0.0 0.702702702703
        cnt_adID (2284,) 0.0 554534.0
        positionID (7219,) 1 7645
        hometown_p (35,) 0 34
        action_installed (2,) 0 1
        creativeID (6315,) 1 6582
        cnt_creativeID (2575,) 0.0 417728.0
        cvt_appPlatform (26,) 0.0235564738109 0.0285316369933
        clickTime_hour (24,) 0 23
        cvt_clickTime_week (8,) 0.0 0.0315482248123
        camgaignID (677,) 1 720
        action_cate_recent (22,) 0 28
        cvt_education (104,) 0.0198213640077 0.0352852852853
        inst_is_installed (2,) 0 1
        cvt_appCategory (149,) 0.0 0.702702702703
        sitesetID (3,) 0 2

        clickTime_day|cvt_userID
        cvt_creativeID|cvt_positionID
        cvt_creativeID|cvt_userID
        cvt_connectionType|cvt_positionID
        cvt_camgaignID|cvt_positionID
        cvt_positionID|cvt_userID
        cvt_camgaignID|cvt_creativeID
        cvt_positionID|cvt_positionID
        cnt_userID|cvt_userID
        cnt_userID|cvt_positionID
        cvt_camgaignID|cvt_userID
        clickTime_day|cvt_advertiserID
        cvt_creativeID|cvt_creativeID
        cnt_connectionType|cvt_userID
        cvt_advertiserID|cvt_connectionType
        cvt_positionID|cvt_sitesetID
        cnt_positionID|cvt_positionID
        cvt_userID|cvt_userID
        cnt_userID|cvt_creativeID
        """

    feature_list = ['label',
                    'cvt_positionID',
                    'cvt_userID',
                    'cvt_creativeID',
                    'cvt_camgaignID',
                    'clickTime_day',
                    'cvt_connectionType',
                    'cnt_userID',
                    'cvt_advertiserID',
                    'cnt_connectionType',
                    'cvt_sitesetID',
                    'action_cate',
                    'tt_cnt_appcate',
                    'cnt_positionID',
                    'appID',
                    'cnt_appID',
                    'appCategory',
                    'cnt_advertiserID',
                    'cvt_adID',
                    'age',
                    'inst_app_installed',
                    'gender',
                    'cvt_gender',
                    'cnt_camgaignID',
                    'cnt_appCategory',
                    'advertiserID',
                    'positionType',
                    'cvt_appID',
                    'cnt_adID',
                    'positionID',
                    'hometown_p',
                    'action_installed',
                    'creativeID',
                    'cnt_creativeID',
                    'cvt_appPlatform',
                    'clickTime_hour',
                    'cvt_clickTime_week',
                    'camgaignID',
                    'action_cate_recent',
                    'cvt_education',
                    'inst_is_installed',
                    'cvt_appCategory',
                    'sitesetID',
                    'instanceID',
                    'clickTime_day_cvt_userID',
                    'cvt_creativeID_cvt_positionID',
                    'cvt_creativeID_cvt_userID',
                    'cvt_connectionType_cvt_positionID',
                    'cvt_camgaignID_cvt_positionID',
                    'cvt_positionID_cvt_userID',
                    'cvt_camgaignID_cvt_creativeID',
                    'cvt_positionID_cvt_positionID',
                    'cnt_userID_cvt_userID',
                    'cnt_userID_cvt_positionID',
                    'cvt_camgaignID_cvt_userID',
                    'clickTime_day_cvt_advertiserID',
                    'cvt_creativeID_cvt_creativeID',
                    'cnt_connectionType_cvt_userID',
                    'cvt_advertiserID_cvt_connectionType',
                    'cvt_positionID_cvt_sitesetID',
                    'cnt_positionID_cvt_positionID',
                    'cvt_userID_cvt_userID',
                    'cnt_userID_cvt_creativeID'
                    ]
    need_OHE_list = ['clickTime_day',
                     'action_cate',
                     'appID',
                     'appCategory',
                     'age',
                     'gender',
                     'advertiserID',
                     'positionType'
                     'positionID',
                     'hometown_p',
                     'action_installed',
                     'creativeID',
                     'clickTime_hour',
                     'camgaignID',
                     'inst_is_installed',
                     'sitesetID',
                     'clickTime_day_cvt_userID'
                     'clickTime_day_cvt_advertiserID']
    LR_x = pd.concat([df_train, df_pre], axis=0)
    LR_x = LR_x.loc[:, feature_list]
    # LR_x.cnt_userID = LR_x.cnt_userID / 10000
    # LR_x.cnt_connectionType = LR_x.cnt_connectionType / 10000
    # LR_x.cnt_positionID = LR_x.cnt_positionID / 10000
    # LR_x.cnt_appID = LR_x.cnt_appID / 10000
    # LR_x.cnt_advertiserID = LR_x.cnt_advertiserID / 10000
    # LR_x.cnt_camgaignID = LR_x.cnt_camgaignID / 10000
    # LR_x.cnt_appCategory = LR_x.cnt_appCategory / 10000
    # LR_x.cnt_adID = LR_x.cnt_adID / 10000
    # LR_x.cnt_creativeID = LR_x.cnt_creativeID / 10000

    LR_x['appID'] = pd.Series(
        LR_x['appID']).astype('category').values.codes
    LR_x['appCategory'] = pd.Series(
        LR_x['appCategory']).astype('category').values.codes
    LR_x.age = LR_x.age / 5
    LR_x['inst_app_installed'] = pd.Series(
        LR_x['inst_app_installed']).astype('category').values.codes
    LR_x['advertiserID'] = pd.Series(
        LR_x['advertiserID']).astype('category').values.codes
    LR_x['positionID'] = pd.Series(
        LR_x['positionID']).astype('category').values.codes
    LR_x['creativeID'] = pd.Series(
        LR_x['creativeID']).astype('category').values.codes
    LR_x['camgaignID'] = pd.Series(
        LR_x['camgaignID']).astype('category').values.codes

    LR_x['clickTime_day_cvt_userID'] = pd.Series(LR_x['clickTime_day'].astype(
        str) + (LR_x['cvt_userID'] / 0.05).astype(str)).astype('category').values.codes
    LR_x['cvt_creativeID_cvt_positionID'] = LR_x['cvt_creativeID'] + \
        LR_x['cvt_positionID']
    LR_x['cvt_creativeID_cvt_userID'] = LR_x['cvt_creativeID'] + \
        LR_x['cvt_userID']
    LR_x['cvt_connectionType_cvt_positionID'] = LR_x['cvt_connectionType'] + \
        LR_x['cvt_positionID']
    LR_x['cvt_camgaignID_cvt_positionID'] = LR_x['cvt_camgaignID'] + \
        LR_x['cvt_positionID']
    LR_x['cvt_positionID_cvt_userID'] = LR_x['cvt_positionID'] + \
        LR_x['cvt_userID']
    LR_x['cvt_camgaignID_cvt_creativeID'] = LR_x['cvt_camgaignID'] + \
        LR_x['cvt_creativeID']
    LR_x['cvt_positionID_cvt_positionID'] = LR_x['cvt_positionID'] + \
        LR_x['cvt_positionID']
    LR_x['cnt_userID_cvt_userID'] = LR_x[
        'cnt_userID'] * LR_x['cvt_userID']
    LR_x['cnt_userID_cvt_positionID'] = LR_x['cnt_userID'] * \
        LR_x['cvt_positionID']
    LR_x['cvt_camgaignID_cvt_userID'] = LR_x['cvt_camgaignID'] + \
        LR_x['cvt_userID']
    LR_x['cnt_userID_cvt_userID'] = LR_x[
        'cnt_userID'] * LR_x['cvt_userID']
    LR_x['clickTime_day_cvt_advertiserID'] = pd.Series(LR_x['clickTime_day'].astype(
        str) + (LR_x['cvt_advertiserID'] / 0.05).astype(str)).astype('category').values.codes

    LR_x['cvt_advertiserID_cvt_connectionType'] = LR_x['cvt_advertiserID'] + \
        LR_x['cvt_connectionType']
    LR_x['cvt_positionID_cvt_sitesetID'] = LR_x['cvt_positionID'] + \
        LR_x['cvt_sitesetID']
    LR_x['cnt_positionID_cvt_positionID'] = LR_x['cnt_positionID'] * \
        LR_x['cvt_positionID']

    LR_x['cvt_userID_cvt_userID'] = LR_x[
        'cvt_userID'] + LR_x['cvt_userID']
    LR_x['cnt_userID_cvt_creativeID'] = LR_x['cnt_userID'] * \
        LR_x['cvt_creativeID']

    LR_x['instanceID'].fillna(-1, inplace=True)
    LR_x.fillna(0, inplace=True)

    for feature in feature_list:
        if feature not in need_OHE_list + ['instanceID', 'label']:
            LR_x[feature] = StandardScaler().fit_transform(LR_x[feature])

    pre_x = LR_x.loc[LR_x['instanceID'] > 0].copy()
    pre_x.sort_values('instanceID', inplace=True)
    inst_id = pre_x['instanceID'].copy().values
    pre_x.drop(['instanceID'], axis=1, inplace=True)
    pre_x.drop(['label'], axis=1, inplace=True)
    LR_x.drop(['instanceID'], axis=1, inplace=True)

    encoder = None
    enc_LR_x = LR_x.drop('label', axis=1)
    if with_ohe and need_OHE_list is not None:
        idx_to_ohe = [i for i, j in enumerate(
            enc_LR_x.columns) if j in need_OHE_list]
        encoder = OneHotEncoder(categorical_features=idx_to_ohe)
        encoder.fit(enc_LR_x.values)

    if not if_CV:
        train_x = LR_x.loc[LR_x['clickTime_day'] <= (30 - test_days), :].copy()
        test_x = LR_x.loc[(LR_x['clickTime_day'] > (30 - test_days))
                          & (LR_x['clickTime_day'] <= 30), :].copy()
        print 'pre_x.shape:'
        print pre_x.shape

        print train_x.columns
        train_y = np.round(train_x['label']).astype(int).values
        train_x.drop('label', 1, inplace=True)

        test_y = np.round(test_x['label']).astype(int).values
        test_x.drop('label', 1, inplace=True)

        if with_ohe and need_OHE_list is not None:
            train_x = encoder.transform(train_x)
            test_x = encoder.transform(test_x)
            pre_x = encoder.transform(pre_x)
            pickle.dump(train_x, open('train_x.pkl', 'wb'), 2)
            pickle.dump(test_x, open('test_x.pkl', 'wb'), 2)
            pickle.dump(train_y, open('train_y.pkl', 'wb'), 2)
            pickle.dump(test_y, open('test_y.pkl', 'wb'), 2)
            pickle.dump(pre_x, open('pre_x.pkl', 'wb'), 2)
        else:
            train_x = train_x.values
            test_x = test_x.values
            pre_x = pre_x.values

        return train_x, train_y, test_x, test_y, pre_x, inst_id
    else:
        x = LR_x.loc[LR_x['clickTime_day'] <= 30, :].copy()
        groups = x.clickTime_day.values
        y = np.round(x['label']).astype(int).values
        x.drop('label', 1, inplace=True)

        if with_ohe and need_OHE_list is not None:
            x = encoder.transform(x)
            pre_x = encoder.transform(pre_x)
        else:
            x = x.values
            pre_x = pre_x.values

        return x, y, groups, pre_x, inst_id