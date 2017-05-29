# coding: utf-8
# pylint: disable=C0103, C0111

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
#from pyspark.ml.feature import OneHotEncoder as OHE
#import pyspark.sql.functions as F
import numpy as np

from data import *
import pdb
import pickle

#from tqdm import tqdm
import os


def to_LGBM(df_train, df_pre, test_days=2):
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
                    'positionType'
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
                    'sitesetID'
                    ]

    inst_id = df_pre['instanceID'].values
    df_pre.drop(['instanceID'], axis=1, inplace=True)
    LGBM_x = pd.concat([df_train, df_pre], axis=0)
    LGBM_x = LGBM_x.ix[:, feature_list]
    LGBM_x.cnt_userID = LGBM_x.cnt_userID / 10000
    LGBM_x.cnt_connectionType = LGBM_x.cnt_connectionType / 10000
    LGBM_x.cnt_positionID = LGBM_x.cnt_positionID / 10000
    LGBM_x.cnt_appID = LGBM_x.cnt_appID / 10000
    LGBM_x.cnt_advertiserID = LGBM_x.cnt_advertiserID / 10000
    LGBM_x.cnt_camgaignID = LGBM_x.cnt_camgaignID / 10000
    LGBM_x.cnt_appCategory = LGBM_x.cnt_appCategory / 10000
    LGBM_x.cnt_adID = LGBM_x.cnt_adID / 10000
    LGBM_x.cnt_creativeID = LGBM_x.cnt_creativeID / 10000
    LGBM_x['action_cate'] = pd.Series(
        LGBM_x['action_cate']).astype('category').values.codes
    LGBM_x['appID'] = pd.Series(
        LGBM_x['appID']).astype('category').values.codes
    LGBM_x['appCategory'] = pd.Series(
        LGBM_x['appCategory']).astype('category').values.codes
    LGBM_x.age = LGBM_x.age / 5
    LGBM_x['inst_app_installed'] = pd.Series(
        LGBM_x['inst_app_installed']).astype('category').values.codes
    LGBM_x['advertiserID'] = pd.Series(
        LGBM_x['advertiserID']).astype('category').values.codes
    LGBM_x['positionID'] = pd.Series(
        LGBM_x['positionID']).astype('category').values.codes
    LGBM_x['creativeID'] = pd.Series(
        LGBM_x['creativeID']).astype('category').values.codes
    LGBM_x['camgaignID'] = pd.Series(
        LGBM_x['camgaignID']).astype('category').values.codes
    LGBM_x['clickTime_day_cvt_userID'] = pd.Series(LGBM_x['clickTime_day'].astype(
        'string') + (LGBM_x['cvt_userID'] / 0.05).astype('string')).astype('category').values.codes
    LGBM_x['cvt_creativeID_cvt_positionID'] = LGBM_x['cvt_creativeID'] + \
        LGBM_x['cvt_positionID']
    LGBM_x['cvt_creativeID_cvt_userID'] = LGBM_x['cvt_creativeID'] + \
        LGBM_x['cvt_userID']
    LGBM_x['cvt_connectionType_cvt_positionID'] = LGBM_x['cvt_connectionType'] + \
        LGBM_x['cvt_positionID']
    LGBM_x['cvt_camgaignID_cvt_positionID'] = LGBM_x['cvt_camgaignID'] + \
        LGBM_x['cvt_positionID']
    LGBM_x['cvt_positionID_cvt_userID'] = LGBM_x['cvt_positionID'] + \
        LGBM_x['cvt_userID']
    LGBM_x['cvt_camgaignID_cvt_creativeID'] = LGBM_x['cvt_camgaignID'] + \
        LGBM_x['cvt_creativeID']
    LGBM_x['cvt_positionID_cvt_positionID'] = LGBM_x['cvt_positionID'] + \
        LGBM_x['cvt_positionID']
    LGBM_x['cnt_userID_cvt_userID'] = LGBM_x['cnt_userID'] * LGBM_x['cvt_userID']
    LGBM_x['cnt_userID_cvt_positionID'] = LGBM_x['cnt_userID'] * \
        LGBM_x['cvt_positionID']
    LGBM_x['cvt_camgaignID_cvt_userID'] = LGBM_x['cvt_camgaignID'] + \
        LGBM_x['cvt_userID']
    LGBM_x['cnt_userID_cvt_userID'] = LGBM_x['cnt_userID'] * LGBM_x['cvt_userID']
    LGBM_x['clickTime_day_cvt_advertiserID'] = pd.Series(LGBM_x['clickTime_day'].astype(
        'string') + (LGBM_x['cvt_advertiserID'] / 0.05).astype('string')).astype('category').values.codes

    LGBM_x['cvt_advertiserID_cvt_connectionType'] = LGBM_x['cvt_advertiserID'] + \
        LGBM_x['cvt_connectionType']
    LGBM_x['cvt_positionID_cvt_sitesetID'] = LGBM_x['cvt_positionID'] + \
        LGBM_x['cvt_sitesetID']
    LGBM_x['cnt_positionID_cvt_positionID'] = LGBM_x['cnt_positionID'] * \
        LGBM_x['cvt_positionID']

    LGBM_x['cvt_userID_cvt_userID'] = LGBM_x['cvt_userID'] + LGBM_x['cvt_userID']
    LGBM_x['cnt_userID_cvt_creativeID'] = LGBM_x['cnt_userID'] * \
        LGBM_x['cvt_creativeID']

    pre_x = LGBM_x.iloc[-df_pre.shape[0]:].copy()
    print 'pre x columns: '
    print pre_x.columns
    train_x = LGBM_x.ix[LGBM_x['clickTime_day'] <= (30 - test_days), :].copy()
    test_x = LGBM_x.ix[(LGBM_x['clickTime_day'] > (30 - test_days))
                       & (LGBM_x['clickTime_day'] <= 30), :].copy()
    print 'pre_x.shape:'
    print pre_x.shape

    print train_x.columns
    train_y = np.round(train_x['label']).astype(int).values
    train_x.drop('label', 1, inplace=True)

    test_y = np.round(test_x['label']).astype(int).values
    test_x.drop('label', 1, inplace=True)

    pre_x.drop(['label'], axis=1, inplace=True)

    train_x = train_x.values
    test_x = test_x.values
    pre_x = pre_x.values

    return train_x, train_y, test_x, test_y, pre_x, inst_id


def get_hist_feature(hist_list, df_concat, with_count=True):
    """
    单变量历史转化率，历史数据特征
    :return:
    """
    for vn in hist_list:
        print vn
        df_concat['cvt_' + vn] = np.zeros(df_concat.shape[0])
        if with_count:
            df_concat['cnt_' + vn] = np.zeros(df_concat.shape[0])
        # 第十七天使用当天的妆化率
        for i in range(17, 32):
            print i
            df_concat['key'] = df_concat[vn].astype('category').values.codes
            if i > 17:
                df_grp_ = df_concat.ix[df_concat['clickTime_day'] < i, [
                    'label', 'key', 'conversionTime']].copy()
                df_grp_['conversionTime'].fillna(0)
                df_grp = df_grp_.ix[df_grp_['conversionTime'] / 10000 < i, [
                    'label', 'key']].copy()
                cnt = df_grp.groupby('key').aggregate(np.size)
                sum = df_grp.groupby('key').aggregate(np.sum)
                v_codes = df_concat.ix[df_concat['clickTime_day']
                                       == i, 'key'].values
                if len(list(set(v_codes).intersection(set(cnt.index)))) != 0:
                    _cnt = cnt.loc[v_codes, :].values
                    _sum = sum.loc[v_codes, :].values
                    __cnt = _cnt.copy()
                    __cnt[np.isnan(__cnt)] = 1
                    _cnt[np.isnan(_cnt)] = 0
                    _sum[np.isnan(_sum)] = 0
                    df_concat.ix[df_concat['clickTime_day'] == i,
                                 'cvt_' + vn] = _sum.astype('float64') / __cnt
                    if with_count:
                        df_concat.ix[df_concat['clickTime_day']
                                     == i, 'cnt_' + vn] = _cnt
            # else:
            #     df_grp = df_concat.ix[df_concat['clickTime_day'] == i, [
            #         'label', 'key']].copy()

    df_concat.drop(['key'], axis=1, inplace=True)
    df_concat[np.isnan(df_concat)] = 0
    # df_concat.to_csv('hist_feature.csv', index=False)


def multi_hist():
    """
    多个变量的历史转化率
    :return:
    """
    pass


def get_feature(for_train=True):
    """
    提取相关特征 
    :name, unique.shape, min, max
    userID (2805118,) 1 2805118                   
    creativeID (6582,) 1 6582
    positionID (7645,) df_sum 7645
    adID (3616,) 1 3616
    camgaignID (720,) 1 720
    advertiserID (91,) 1 91 
    appID (50,) 14 472
    sitesetID (3,) 0 2

    appCategory (14,) 0 503
    appPlatform (2,) 1 2
    education (8,) 0 7
    gender (3,) 0 2
    haveBaby (7,) 0 6
    marriageStatus (4,) 0 3
    positionType (6,) 0 5
    hometown_c (22,) 0 21
    hometown_p (35,) 0 34
    residence_c (22,) 0 21
    residence_p (35,) 0 34
    telecomsOperator (4,) 0 3
    connectionType (5,) 0 4
    clickTime_week (7,) 0 6

    age (81,) 0 80
    action_cate (88,) 0 117
    action_cate_recent (22,) 0 28
    action_installed (2,) 0 1
    inst_app_installed (13,) 0 282777
    inst_cate_percent (3679,) 0.0 1.0
    inst_cnt_appcate (89,) 0 172
    inst_cnt_installed (371,) 0 505
    inst_is_installed (2,) 0 1
    tt_cnt_appcate (123,) 0 250
    tt_is_installed (2,) 0 1
    clickTime_day (15,) 17 31
    clickTime_hour (24,) 0 23
    clickTime_minute (60,) 0 59

    cvt_userID (51,) 0.0 1.0
    cnt_userID (87,) 0.0 111.0
    cvt_creativeID (3880,) 0.0 1.0
    cnt_creativeID (2575,) 0.0 417728.0
    cvt_positionID (5454,) 0.0 1.0
    cnt_positionID (3059,) 0.0 351373.0
    cvt_adID (3267,) 0.0 1.0
    cnt_adID (2284,) 0.0 554534.0
    cvt_camgaignID (2377,) 0.0 1.0
    cnt_camgaignID (2109,) 0.0 554534.0
    cvt_advertiserID (574,) 0.0 0.702702702703
    cnt_advertiserID (625,) 0.0 1765710.0
    cvt_appID (399,) 0.0 0.702702702703
    cnt_appID (443,) 0.0 1765710.0
    cvt_sitesetID (40,) 0.0 0.0476605066562
    cnt_sitesetID (40,) 0.0 2420202.0
    cvt_appCategory (149,) 0.0 0.702702702703
    cnt_appCategory (161,) 0.0 1765710.0
    cvt_appPlatform (27,) 0.0 0.0285316369933
    cnt_appPlatform (27,) 0.0 2654112.0
    cvt_education (105,) 0.0 0.0352852852853
    cnt_education (105,) 0.0 903347.0
    cvt_gender (40,) 0.0 0.0309996027044
    cnt_gender (40,) 0.0 1659511.0
    cvt_haveBaby (92,) 0.0 0.0416462518373
    cnt_haveBaby (92,) 0.0 2733585.0
    cvt_marriageStatus (53,) 0.0 0.0293171931152
    cnt_marriageStatus (53,) 0.0 1375565.0
    cvt_positionType (79,) 0.0 0.26247689464
    cnt_positionType (79,) 0.0 2233082.0
    cvt_hometown_c (287,) 0.0 0.0454545454545
    cnt_hometown_c (285,) 0.0 1251320.0
    cvt_hometown_p (446,) 0.0 0.0363182167563
    cnt_hometown_p (455,) 0.0 1251320.0
    cvt_residence_c (287,) 0.0 0.0354330708661
    cnt_residence_c (287,) 0.0 717383.0
    cvt_residence_p (453,) 0.0 0.0588235294118
    cnt_residence_p (455,) 0.0 314346.0
    cvt_telecomsOperator (53,) 0.0 0.0358006448146
    cnt_telecomsOperator (53,) 0.0 1894971.0
    cvt_connectionType (66,) 0.0 0.0317665577059
    cnt_connectionType (66,) 0.0 2726052.0
    cvt_clickTime_week (8,) 0.0 0.0315482248123
    cnt_clickTime_week (8,) 0.0 325921.0

    """
    if for_train:
        df_file = read_as_pandas(FILE_TRAIN)

    else:
        df_file = read_as_pandas(FILE_TEST)

    not_ohe = []
    to_drop = []

    # creative realated
    # creativeID,adID,camgaignID,advertiserID,appID,appPlatform
    df_ad = read_as_pandas(FILE_AD)
    df_app_category = read_as_pandas(FILE_APP_CATEGORIES)
    df_result = pd.merge(df_file, df_ad, how='left', on='creativeID')
    df_result = pd.merge(df_result, df_app_category, how='left', on='appID')

    # user realated
    df_user = read_as_pandas(FILE_USER)
    # df_user['age'] = np.round(df_user['age'] / 10).astype(int)  # 年龄段
    df_user['hometown_p'] = np.round(
        df_user['hometown'].astype(int) / 100).astype(int)  # 取省份
    df_user['hometown_c'] = np.round(
        df_user['hometown'].astype(int) % 100).astype(int)  # 城市
    df_user['residence_p'] = np.round(
        df_user['residence'].astype(int) / 100).astype(int)  # 取省份
    df_user['residence_c'] = np.round(
        df_user['residence'].astype(int) % 100).astype(int)  # 城市
    df_result = pd.merge(df_result, df_user, how='left', on='userID')

    to_drop += [
        'hometown',
        'residence',
    ]

    # position related
    df_position = read_as_pandas(FILE_POSITION)
    df_result = pd.merge(df_result, df_position, how='left', on='positionID')

    # installed app related
    # 用户已安装列表是否存在该应用、同类应用的数量、所占比例、该用户已经安装app的数量
    df_installed = read_as_pandas(FILE_USER_INSTALLEDAPPS)
    df_group_cnt = df_installed.groupby('userID').count().rename(
        columns={'appID': 'inst_cnt_installed'}).reset_index()
    df_installed_cate = pd.merge(
        df_installed, df_app_category, how='left', on='appID')
    df_group_cnt_user_appcate = df_installed_cate.groupby(['userID', 'appCategory']).count().reset_index().rename(
        columns={'appID': 'inst_cnt_appcate'})
    df_percent = pd.merge(df_group_cnt_user_appcate, df_group_cnt, how='left',
                          on='userID')  # userID, appCategory, inst_cnt_installed, inst_cnt_appcate
    df_percent.fillna(0, inplace=True)
    df_percent['inst_cate_percent'] = df_percent['inst_cnt_appcate'].astype(float) / df_percent[
        'inst_cnt_installed']  # inst_cate_percent
    df_result = pd.merge(df_result, df_percent, how='left',
                         on=['userID', 'appCategory'])
    df_result['inst_cate_percent'].fillna(0, inplace=True)  # 同类应用比例
    df_result['inst_cnt_installed'].fillna(0, inplace=True)

    df_installed['count'] = np.ones(df_installed.shape[0])
    df_group_exist = df_installed.groupby(['userID', 'appID']).count().rename(
        columns={'count': 'inst_is_installed'}).reset_index()
    df_result = pd.merge(df_result, df_group_exist,
                         how='left', on=['userID', 'appID'])
    df_result['inst_is_installed'].fillna(0, inplace=True)
    del df_installed['count']

    df_group_app = df_installed.groupby('appID').count().rename(
        columns={'userID': 'inst_app_installed'}).reset_index()
    df_result = pd.merge(df_result, df_group_app, on='appID', how='left')
    df_result['inst_app_installed'].fillna(0, inplace=True)

    df_result['inst_cnt_installed'] = df_result['inst_cnt_installed'].astype(
        int)  # 用戶已經安裝的app個數
    df_result['inst_is_installed'] = df_result['inst_is_installed'].astype(
        int)  # 該app被安裝的次数
    df_result['inst_cnt_appcate'] = df_result['inst_cnt_appcate'].fillna(
        0).astype(int)  # 同类应用个数
    df_result['inst_app_installed'] = df_result['inst_app_installed'].astype(
        int)  # 该app被安装的次数

    # 安裝流水中是否存在该应用
    df_actions = read_as_pandas(FILE_USER_APP_ACTIONS)
    df_result['index'] = df_result.index
    df_merge = pd.merge(df_actions, df_app_category, how='left', on='appID').rename(
        columns={'appID': 'action_appID', 'appCategory': 'action_appCategory'})
    df_merged = pd.merge(df_result, df_merge, on=['userID'], how='left')
    df_merged['action_installed'] = (df_merged['clickTime'] > df_merged['installTime']) \
        & (df_merged['action_appID'] == df_merged['appID'])

    df_merged['action_cate'] = (df_merged['clickTime'] > df_merged['installTime']) \
        & (df_merged['appCategory'] == df_merged['action_appCategory'])

    df_merged['action_cate_recent'] = (df_merged['clickTime'] > df_merged['installTime']) \
        & (df_merged['clickTime'] - df_merged['installTime'] < 10000) \
        & (df_merged['appCategory'] == df_merged['action_appCategory'])  # 最近两天同类

    df_merged['action_installed'] = df_merged['action_installed'].astype(int)
    df_merged['action_cate'] = df_merged['action_cate'].astype(int)
    df_merged['action_cate_recent'] = df_merged['action_cate_recent'].astype(
        int)
    df_sum = pd.DataFrame(df_merged.loc[:, ['index', 'action_installed', 'action_cate', 'action_cate_recent']]) \
        .groupby(['index']).sum().reset_index()

    df_result['action_installed'] = df_sum['action_installed'].fillna(
        0)  # 用户安装该app的次数
    df_result['action_cate'] = df_sum['action_cate'].fillna(0)  # 用户安装该类别的次数
    df_result['action_cate_recent'] = df_sum['action_cate_recent'].fillna(
        0)  # 用户最近两天安装该类别app的次数

    # 修正已安装数据
    df_result['tt_is_installed'] = (df_result['inst_is_installed'] > 0) \
        & (df_result['action_installed'] > 0)  # clickTime之前是否已经安装过

    df_result['tt_is_installed'] = df_result['tt_is_installed'].astype(int)
    df_result['tt_cnt_appcate'] = df_result['action_cate'] + \
        df_result['inst_cnt_appcate']  # clickTime之前该app同类应用安装次数

    # context
    df_result['clickTime_day'] = pd.Series(
        df_result['clickTime'].astype(str).str.slice(0, 2)).astype(int)
    df_result['clickTime_hour'] = pd.Series(
        df_result['clickTime'].astype(str).str.slice(2, 4)).astype(int)
    df_result['clickTime_minute'] = pd.Series(
        df_result['clickTime'].astype(str).str.slice(4, 6)).astype(int)

    df_result['clickTime_week'] = pd.Series(
        np.floor(df_result['clickTime'].astype(int) / 10000) % 7).astype(int)

    # history pcvr 没考虑时间
    hist_list = [
        'userID',
        'creativeID',
        'positionID',
        'adID',
        'camgaignID',
        'advertiserID',
        'appID',
    ]

    # [17-30] 第一天使用均值替代或使用后一天替代 注意：后4-5天的转化率有些不准(广告商可能没有反馈过来)

    # remove unrelated
    to_drop += ['clickTime', 'index', ]
    not_ohe += [
        'userID',
        'inst_cnt_appcate',
        'inst_cnt_installed',
        'inst_cate_percent',
        'inst_is_installed',
        'inst_app_installed',
        'action_installed',
        'action_cate',
        'action_cate_recent',
        'tt_is_installed',
        'tt_cnt_appcate',
        'clickTime_day',
        'clickTime_hour',
        'clickTime_minute',
    ]

    # if for_train:
    #     to_drop += ['conversionTime']

    df_result.drop(to_drop, axis=1, inplace=True)
    print df_result.columns
    print df_result.head(5)
    return df_result, not_ohe


def get_tf_feature(with_ohe=True, save=True, needDF=False, modelType='LGBM', test_days=2):
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print '重新生成特徵'
        df_train, not_ohe = get_feature(True)
        df_test, not_ohe = get_feature(False)
        shuffle(df_train)
        df_train.fillna(0, inplace=True)
        df_test.fillna(0, inplace=True)

        df_concate = pd.concat([df_train, df_test], axis=0)
        get_hist_feature(['userID',
                          'creativeID',
                          'positionID',
                          'adID',
                          'camgaignID',
                          'advertiserID',
                          'appID',
                          'sitesetID',
                          'appCategory',
                          'appPlatform',
                          'education',
                          'gender',
                          'haveBaby',
                          'marriageStatus',
                          'positionType',
                          'hometown_c',
                          'hometown_p',
                          'residence_c',
                          'residence_p',
                          'telecomsOperator',
                          'connectionType',
                          'clickTime_week'], df_concat=df_concate)
        for column in df_concate.columns:
            print column, df_concate[column].unique().shape, df_concate[column].min(), df_concate[column].max()
        df_train = (df_concate.iloc[:df_train.shape[0], :]).drop(
            ['instanceID'], axis=1)  # 重新赋值
        df_test = df_concate.iloc[-df_test.shape[0]:, :]  # 重新赋值
        if save:
            df_train.to_csv('train.csv', index=False)
            df_test.to_csv('test.csv', index=False)
    else:
        print '從文件加載特徵'
        df_train, not_ohe, df_test = pd.read_csv(
            'train.csv'), None, pd.read_csv('test.csv')

    if needDF:
        return df_train, df_test

    if modelType == 'LGBM':
        train_x, train_y, test_x, test_y, pre_x, inst_id = to_LGBM(
            df_train, df_test, test_days=test_days)

    if with_ohe and not_ohe is not None:
        idx_to_ohe = [i for i, j in enumerate(columns) if j not in not_ohe]
        encoder = OneHotEncoder(categorical_features=idx_to_ohe)
        df_concate = pd.concat([df_train, df_test], axis=1)
        df_concate.fillna(-1)
        encoder.fit(df_concate.values)

        train_x = encoder.transform(train_x)
        test_x = encoder.transform(test_x)
    if not_ohe is None and with_ohe:
        print 'not ohe is None , not going to ohe'

    print train_x.shape, type(train_x), train_y.shape, type(train_y)

    # if save:
    #     pickle.dump(train_x, open('train_x.pkl', 'wb'), 2)
    #     pickle.dump(train_y, open('train_y.pkl', 'wb'), 2)

    #     pickle.dump(test_x, open('test_x.pkl', 'wb'), 2)
    #     pickle.dump(inst_id, open('inst_id.pkl', 'wb'), 2)

    return train_x, train_y, test_x, test_y, pre_x, inst_id


def to_ffm():
    pass


def load_feature(from_file=True, with_ohe=True, modelType='LGBM', test_days=2):
    """
    从文件加载或者。。。
    """
    if from_file:
        filenames = ['train_x.pkl', 'train_y.pkl', 'test_x.pkl', 'inst_id.pkl']
        objs = [pickle.load(open(f, 'rb')) for f in filenames]
        return objs
    else:
        return get_tf_feature(with_ohe=with_ohe, modelType=modelType, test_days=test_days)


def split_train_test(x, y, test_size=0.2, stratify=True, with_df=False):
    """
    分割数据
    :param x:
    :param y:
    :param test_size:
    :param stratify: 考虑不平衡问题
    :return:
    """
    if not with_df:
        if stratify:
            return train_test_split(x, y, test_size=test_size)
        else:
            return train_test_split(x, y, test_size=test_size, stratify=y)

    else:
        if stratify:
            train_x, test_x, train_y, test_y = train_test_split(x.drop(['label'], axis=1).values,
                                                                x['label'].values,
                                                                test_size=test_size,
                                                                stratify=y)
        else:
            train_x, test_x, train_y, test_y = train_test_split(x.drop(['label'], axis=1).values,
                                                                x['label'].values,
                                                                test_size=test_size)

        def to_df(train_x, train_y, x):
            """
            把结果拼接成DataFrame
            :param train_x:
            :param train_y:
            :param x:
            :return:
            """
            _df_train_x = pd.DataFrame(train_x)
            _df_train_x.columns = x.drop(['label'], axis=1).columns
            _df_train_x['label'] = train_y
            return _df_train_x

        return to_df(train_x, train_y, x), to_df(test_x, test_y, x)


def split_train_test_by_day(x, y, test_day_size=1):
    """
    按照天数分割数据
    :param x:
    :param y:
    :param test_day_size: 最后几天作为test
    :return:
    """
    train_x = x.ix[x['clickTime_day'] <= 30 - test_day_size, :].copy()
    test_x = x.ix[x['clickTime_day'] > 30 - test_day_size, :].copy()
    return train_x, test_x


if __name__ == '__main__':
    # df = get_feature(False)
    # print df.head(5)
    # load_feature(from_file=True, with_ohe=False)
    # get_tf_feature(with_ohe=False)
    get_hist_feature(['creativeID'], with_count=True)
