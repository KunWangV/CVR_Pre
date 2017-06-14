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
# from pyspark.ml.feature import OneHotEncoder as OHE
# import pyspark.sql.functions as F
import numpy as np

from data import *
import pdb
import pickle

# from tqdm import tqdm
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def transform_fillna(df, columns=[], method='auto', value=None):
    """
    填充非零值
    :param df: pandas DataFrame
    :param columns: 列名称 list
    :param method: 使用的方法 if value is not None use value, else: method = mean|median ... pandas supported
    :param value: 制定的值
    :return:
    """
    for _c in columns:
        print 'fillna transform, method {}, value {}'.format(method, value)
        if value is not None:
            df[_c] = df[_c].fillna(value)

        else:
            if method == 'auto':
                _type = df[_c].dtype
                if _type == np.int32 or _type == np.int64:
                    df[_c] = df[_c].fillna(0)
                else:
                    df[_c] = df[_c].fillna(df[_c].mean())

            elif method == 'mean':
                df[_c] = df[_c].fillna(df[_c].mean())

            elif method == 'median':
                df[_c] = df[_c].fillna(df[_c].median())

            else:
                pass


def transform_category(df, columns=[]):
    """
    转换成category类别
    :param df: pandas DataFrame
    :param columns: 要转化的列
    :return: None inplace operation
    """
    for _c in columns:
        print 'category transform category', _c
        df[_c] = df[_c].astype('category').values.codes


def transform_log(df, columns=[], threshold=2):
    """
    if m > threshold:
        m = log(m)^2

    这是一个inplace 操作
    :param df: pandas DataFrame
    :param columns: 需要变换的列名
    :param threshold 阈值
    :return: nothing
    """
    for c in columns:
        print 'log transform columns', c
        _type = df[c].dtype
        df.loc[df[c] > threshold, c] = np.power(np.log(df[c].values), 2)
        if _type == np.int32 or _type == np.int64:
            df.loc[df[c] > threshold, c] = np.floor(
                df.loc[df[c] > threshold, c])


def transform_bucket(df, column=None, bucket=[], sort=True):
    """
    分段离散化
    :param df:
    :param column:
    :param bucket:
    :param sort:
    :return:
    """
    if column is not None:

        if sort:
            bucket = sorted(bucket)

        for i, _b in enumerate(bucket):
            if i == 0:
                df[df[column] < _b] = i
                continue

            df[(df[column] < _b) & (df[column] >= bucket[i - 1])] = i

        df[df[column] >= bucket[-1]] = i + 1


def get_cnt_creativeID_positionID():
    """
    cnt_creativeID_positionID
    偷来的特征
    :return:
    """
    train = read_as_pandas(FILE_TRAIN)
    train = train[train['label'] > 0]
    train.drop(
        [
            'label', 'conversionTime', 'userID', 'connectionType',
            'telecomsOperator'
        ],
        axis=1,
        inplace=True)
    tr = train.groupby(['creativeID', 'positionID']).count().rename(
        columns={'clickTime': 'cnt_creativeID_positionID'}).reset_index()
    return tr


def gen_appdi_feature():
    """
    群里说的
    :return:
    """
    train = read_as_pandas(FILE_TRAIN)
    test = read_as_pandas(FILE_TEST)

    user_installedapps = pd.read_csv(FILE_USER_INSTALLEDAPPS)
    user_installedapps = user_installedapps.groupby('userID').agg(
        lambda x: ' '.join(['app' + str(s) for s in x.values])).reset_index()
    df_concate = pd.concat([train, test], axis=0).reset_index(drop=True)
    user_id_all = pd.concat(
        [train.userID, test.userID], axis=0).reset_index(drop=True)
    user_id_all = pd.DataFrame(user_id_all, columns=['userID'])
    train_data = df_concate['clickTime'] // 1000000 <= 28
    eval_data = np.logical_or(df_concate['clickTime'] // 1000000 == 29,
                              df_concate['clickTime'] // 1000000 == 30)
    test_data = df_concate['clickTime'] // 1000000 == 31
    user_installedapps = pd.merge(
        user_id_all.drop_duplicates(),
        user_installedapps,
        on='userID',
        how='left')
    user_installedapps = user_installedapps.fillna('Missing')
    tfv = TfidfVectorizer()
    tfv.fit(user_installedapps.appID)
    print user_installedapps.shape
    # user_installedapps = pd.merge(
    #     user_id_all, user_installedapps, on='userID', how='left')

    print user_id_all[train_data].shape, user_id_all[
        test_data].shape, user_id_all[eval_data].shape
    user_installedapps_train = pd.merge(
        user_id_all[train_data], user_installedapps, on='userID', how='left')
    user_installedapps_test = pd.merge(
        user_id_all[test_data], user_installedapps, on='userID', how='left')
    user_installedapps_eval = pd.merge(
        user_id_all[eval_data], user_installedapps, on='userID', how='left')
    print user_installedapps_train.shape, user_installedapps_eval.shape, user_installedapps_eval.shape

    user_installedapps = user_installedapps.fillna('Missing')
    # user_installedapps_tfv = tfv.transform(user_installedapps.appID)
    train_user_installedapps_tfv = tfv.transform(
        user_installedapps_train.appID)
    eval_user_installedapps_tfv = tfv.transform(user_installedapps_eval.appID)
    test_user_installedapps_tfv = tfv.transform(user_installedapps_test.appID)
    # print user_installedapps_tfv.shape, type(user_installedapps_tfv)
    # np_user_tfv = np.concatenate((user_installedapps.userID.values, np.asarray(user_installedapps_tfv)), axis=1)
    # print np_user_tfv.shape, type(np_user_tfv)
    # print type(user_installedapps_tfv)
    print train_user_installedapps_tfv.shape, eval_user_installedapps_tfv.shape, test_user_installedapps_tfv.shape
    return train_user_installedapps_tfv, eval_user_installedapps_tfv, test_user_installedapps_tfv
    # pickle.dump(np_user_tfv, open('user_installedapps_tfv.pkl', 'w'), 2)
    # pickle.dump(user_installedapps_tfv, open('installedapps_tfv.pkl', 'w'), 2)


def get_click_features(_df, distinct=True):
    """
    提取点击次数特征
    :param df:
    :param distinct: 去重
    :return:
    """
    # add
    print _df.shape
    _df['mindex'] = range(_df.shape[0])
    columns_orgin = _df.columns
    columns_add = [
        'user_day_click_cnt',  # 该用户当天的点击次数
        'user_cri_day_click_cnt',  # 用户广告当天的点击数
        'cri_day_click_cnt',  # 广告没填的点击数
        'cri_uuser_click_cnt',  # 点击该广告的用户数
        'rpt_click_cnt',  # 当天重复点击此次数
        'is_rpt_first_click',  # 是否是当天的重复点击的第一次点击
        'is_rpt_last_click',  # 是否是当天重复点击的最后一次点击
        'position_day_click_cnt',  # 该位置每天的点击次数
        'postion_cri_day_click_cnt',  # 该位置 广告的日点击率
        'user_app_day_click_cnt',
        'app_day_click_cnt',
        'app_uuser_click_cnt',
        'postion_app_day_click_cnt',
        'is_day_rpt_first_click',
        'is_day_rpt_last_click',
        'rpt_day_click_cnt',
    ]
    for _c in columns_add:
        _df[_c] = np.zeros(_df.shape[0])

    days = pd.Series(np.floor(_df['clickTime'] / 1000000)).unique().astype(int)
    print 'days to caculdate: ', days
    _df['clickTime_Day'] = np.floor(_df['clickTime'] / 1000000).astype(int)

    df_pre = None

    for day in days:
        print 'day {} '.format(day)
        df = _df.loc[_df['clickTime_Day'] == day, columns_orgin].copy()
        df_app = read_as_pandas(FILE_AD).loc[:, ['creativeID', 'adID']]
        df = pd.merge(df, df_app, how='left', on='creativeID')
        print df.shape

        # 该user该当天总的点击次数
        # user_day_click_cnt
        grp_columns = ['userID']
        df_grp = df.groupby(grp_columns).count().reset_index().rename(
            columns={'mindex': 'user_day_click_cnt'
                     }).loc[:, grp_columns + ['user_day_click_cnt']]
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        # 该user当天点击该creative的次数
        # user_cri_day_click_cnt
        grp_columns = ['userID', 'creativeID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'user_cri_day_click_cnt'})
            .loc[:, grp_columns + ['user_cri_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['userID', 'adID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'user_app_day_click_cnt'})
            .loc[:, grp_columns + ['user_app_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        # 该广告当天被点击的次数 | 点击该广告的所有user的个数
        # cri_day_click_cnt | cri_uuser_click_cnt
        grp_columns = ['creativeID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'cri_day_click_cnt'})
            .loc[:, grp_columns + ['cri_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['creativeID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).agg(pd.Series.nunique).reset_index()
            .rename(columns={'mindex': 'cri_uuser_click_cnt'
                             }).loc[:, grp_columns + ['cri_uuser_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['adID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'app_day_click_cnt'})
            .loc[:, grp_columns + ['app_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['adID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).agg(pd.Series.nunique).reset_index()
            .rename(columns={'mindex': 'app_uuser_click_cnt'
                             }).loc[:, grp_columns + ['app_uuser_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        # 该positionID的点击率
        grp_columns = ['positionID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'position_day_click_cnt'})
            .loc[:, grp_columns + ['position_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['positionID', 'creativeID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).agg(pd.Series.nunique).reset_index()
            .rename(columns={'mindex': 'postion_cri_day_click_cnt'})
            .loc[:, grp_columns + ['postion_cri_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        grp_columns = ['positionID', 'adID']
        df_grp = pd.DataFrame(
            df.groupby(grp_columns).agg(pd.Series.nunique).reset_index()
            .rename(columns={'mindex': 'postion_app_day_click_cnt'})
            .loc[:, grp_columns + ['postion_app_day_click_cnt']])
        df = pd.merge(df, df_grp, how='left', on=grp_columns)

        # 该user重复点击点击次数 | 是否是第一次 | 是否是最后一次
        # rpt_click_cnt | is_rpt_first_click | is_rpt_last_click
        grp_columns = ['clickTime', 'creativeID', 'userID', 'positionID']
        # grp_columns = ['creativeID', 'userID', 'positionID']
        df_min_grp = pd.DataFrame(
            df.groupby(grp_columns).min().reset_index().rename(
                columns={'mindex': 'rpt_min_index'}).loc[:, grp_columns +
                                                         ['rpt_min_index']])

        df_max_grp = pd.DataFrame(
            df.groupby(grp_columns).max().reset_index().rename(
                columns={'mindex': 'rpt_max_index'}).loc[:, grp_columns +
                                                         ['rpt_max_index']])

        df_cnt_grpu = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'rpt_click_cnt'}).loc[:, grp_columns +
                                                         ['rpt_click_cnt']])

        df = pd.merge(df, df_min_grp, how='left', on=grp_columns)
        df = pd.merge(df, df_max_grp, how='left', on=grp_columns)
        df = pd.merge(df, df_cnt_grpu, how='left', on=grp_columns)
        df['is_rpt_first_click'] = df['rpt_min_index'] == df['mindex']
        df['is_rpt_last_click'] = df['rpt_max_index'] == df['mindex']

        df['is_rpt_first_click'] = df['is_rpt_first_click'].astype(
            int)  # boolean to int
        df['is_rpt_last_click'] = df['is_rpt_last_click'].astype(
            int)  # boolean to int

        df.drop(['rpt_min_index', 'rpt_max_index'], axis=1, inplace=True)

        # grp_columns = ['clickTime', 'creativeID', 'userID', 'positionID']
        grp_columns = ['adID', 'userID', 'positionID']
        df_min_grp = pd.DataFrame(
            df.groupby(grp_columns).min().reset_index().rename(
                columns={'mindex': 'rpt_min_index'}).loc[:, grp_columns +
                                                         ['rpt_min_index']])

        df_max_grp = pd.DataFrame(
            df.groupby(grp_columns).max().reset_index().rename(
                columns={'mindex': 'rpt_max_index'}).loc[:, grp_columns +
                                                         ['rpt_max_index']])

        df_cnt_grpu = pd.DataFrame(
            df.groupby(grp_columns).count().reset_index().rename(
                columns={'mindex': 'rpt_day_click_cnt'})
            .loc[:, grp_columns + ['rpt_day_click_cnt']])

        df = pd.merge(df, df_min_grp, how='left', on=grp_columns)
        df = pd.merge(df, df_max_grp, how='left', on=grp_columns)
        df = pd.merge(df, df_cnt_grpu, how='left', on=grp_columns)
        df['is_day_rpt_first_click'] = df['rpt_min_index'] == df['mindex']
        df['is_day_rpt_last_click'] = df['rpt_max_index'] == df['mindex']

        df['is_day_rpt_first_click'] = df['is_day_rpt_first_click'].astype(
            int)  # boolean to int
        df['is_day_rpt_last_click'] = df['is_day_rpt_last_click'].astype(
            int)  # boolean to int

        df.drop(['rpt_min_index', 'rpt_max_index'], axis=1, inplace=True)
        print df.describe()

        # print df.columns
        if df_pre is None:
            df_pre = df

        else:
            df_pre = pd.concat([df_pre, df], axis=0)

    df_pre.sort_values(['mindex'], inplace=True)
    df_pre.drop(['mindex', 'adID'], axis=1, inplace=True)

    print df_pre.shape
    print df_pre.describe()
    return df_pre

    # df['findex'] = range(df.shape[0])
    # dist_columns = ['clickTime', 'creativeID', 'userID', 'positionID', ]
    #
    # def func_agg(x):
    #     i = x['label'].argmax()  # 索引
    #     if x.loc[i, 'label'] == 1:
    #         dd = pd.DataFrame(x.loc[i].copy())
    #         dd['cnt_dup'] = x.loc[i, 'findex'] - x['findex'].min() + 1  # 点击次数
    #
    #     else:
    #         dd = pd.DataFrame(x.loc[x['findex'].argmax()].copy())
    #         dd['cnt_dup'] = x.shape[0]  # 点击次数
    #
    #     return dd
    #
    # df_grp = df.groupby(dist_columns)
    # df_result = df_grp.apply(func_agg)
    # if distinct:
    #     df_result = df_result.reset_index().drop('findex')
    #     return df_result
    #
    # else:
    #     pass


def get_hist_feature(hist_list,
                     df_concat,
                     with_count=True,
                     with_pre_day_cvt=True,
                     with_pre_day_cnt=True):
    """
    单变量历史转化率，历史数据特征
    :param hist_list:
    :param df_concat:
    :param with_count:
    :param with_pre_day_cvt: 前一天的转化率
    :param with_pre_day_cnt: 前一天的点击个数
    :return:
    """
    for vn in hist_list:
        print '>>>> get hist of', vn
        df_concat['cvt_' + vn] = np.zeros(df_concat.shape[0])
        if with_count:
            df_concat['cnt_' + vn] = np.zeros(df_concat.shape[0])

        if with_pre_day_cnt:
            df_concat['cnt_' + vn] = np.zeros(df_concat.shape[0])

        # 第十七天使用当天的妆化率
        _pre_cnt = None
        _pre_sum = None
        days = df_concat['clickTime_day'].unique().values
        print days
        start = days[1]
        for i in days:
            print 'get hist of {}, day {}'.format(vn, i)
            df_concat['key'] = df_concat[vn].astype('category').values.codes
            if i > start:
                df_grp_ = df_concat.loc[df_concat['clickTime_day'] < i,
                                        ['label', 'key']].copy()

            else:
                df_grp_ = df_concat.loc[df_concat['clickTime_day'] == i,
                                        ['label', 'key']].copy()
            # 当前天
            pre_grp_ = df_concat.loc[df_concat['clickTime_day'] == i,
                                     ['label', 'key']].copy()
            pre_grp = pre_grp_
            pre_cnt = pre_grp.groupby('key').aggregate(np.size)
            pre_sum = pre_grp.groupby('key').aggregate(np.sum)

            # 历史
            df_grp = df_grp_
            cnt = df_grp.groupby('key').aggregate(np.size)
            sum = df_grp.groupby('key').aggregate(np.sum)
            v_codes = df_concat.loc[df_concat['clickTime_day'] == i,
                                    'key'].values

            if len(list(set(v_codes).intersection(set(cnt.index)))) != 0:
                _cnt = cnt.loc[v_codes, :].values
                _sum = sum.loc[v_codes, :].values
                __cnt = _cnt.copy()
                __cnt[np.isnan(__cnt)] = 1
                _cnt[np.isnan(_cnt)] = 0
                _sum[np.isnan(_sum)] = 0
                df_concat.loc[df_concat['clickTime_day'] == i, 'cvt_' +
                              vn] = _sum.astype('float64') / __cnt
                if with_count:
                    df_concat.loc[df_concat['clickTime_day'] == i, 'cnt_' +
                                  vn] = _cnt.astype('float64')

            if _pre_cnt is not None and len(
                    list(set(v_codes).intersection(set(_pre_cnt.index)))) != 0:
                if with_pre_day_cvt and _pre_cnt is not None and _pre_sum is not None:
                    _cnt = _pre_cnt.loc[v_codes, :].values
                    _sum = _pre_sum.loc[v_codes, :].values
                    __cnt = _cnt.copy()
                    __cnt[np.isnan(__cnt)] = 1
                    _cnt[np.isnan(_cnt)] = 0
                    _sum[np.isnan(_sum)] = 0
                    df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cvt_' +
                                  vn] = _sum.astype('float64') / __cnt
                    if with_pre_day_cnt:
                        df_concat.loc[df_concat['clickTime_day'] == i,
                                      'pre_cnt_' + vn] = _cnt.astype('float64')

            elif _pre_cnt is None and _pre_sum is None:  # 第17天的前一天 设置为第17天本身
                df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cvt_' +
                              vn] = df_concat.loc[df_concat['clickTime_day'] ==
                                                  i, 'cvt_' + vn]
                if with_pre_day_cnt:
                    df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cnt_' +
                                  vn] = df_concat.loc[df_concat[
                                      'clickTime_day'] == i, 'cnt_' + vn]

            _pre_cnt = pre_cnt
            _pre_sum = pre_sum

    df_concat.drop(['key'], axis=1, inplace=True)


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
        # df_file = read_as_pandas(FILE_TRAIN)
        df_file = pd.read_hdf('../../train_days27.hdf5')
        # df_file = pd.read_csv('../../train_days27.csv')
    else:
        # df_file = read_as_pandas(FILE_TEST)
        df_file = pd.read_hdf('../../test.hdf5')

    print '== get click feature ===='
    df_file = get_click_features(df_file)

    not_ohe = []
    to_drop = []

    # creative realated
    # creativeID,adID,camgaignID,advertiserID,appID,appPlatform
    print '== get add feature ==='
    df_ad = read_as_pandas(FILE_AD)
    df_app_category = read_as_pandas(FILE_APP_CATEGORIES)
    df_result = pd.merge(df_file, df_ad, how='left', on='creativeID')
    df_result = pd.merge(df_result, df_app_category, how='left', on='appID')

    # user realated
    print '== get user feature ==='
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
    print '== get position feature =='
    df_position = read_as_pandas(FILE_POSITION)
    df_result = pd.merge(df_result, df_position, how='left', on='positionID')

    # installed app related
    # 用户已安装列表是否存在该应用、同类应用的数量、所占比例、该用户已经安装app的数量
    # print '== get install feature =='
    # df_installed = read_as_pandas(FILE_USER_INSTALLEDAPPS)

    # df_group_cnt = df_installed.groupby('userID').count().rename(
    #     columns={'appID': 'inst_cnt_installed'}).reset_index()
    # df_installed_cate = pd.merge(
    #     df_installed, df_app_category, how='left', on='appID')
    # df_group_cnt_user_appcate = df_installed_cate.groupby(['userID', 'appCategory']).count().reset_index().rename(
    #     columns={'appID': 'inst_cnt_appcate'})
    # df_percent = pd.merge(df_group_cnt_user_appcate, df_group_cnt, how='left',
    #                       on='userID')  # userID, appCategory, inst_cnt_installed, inst_cnt_appcate
    # df_percent['inst_cate_percent'] = df_percent['inst_cnt_appcate'].astype(float) / df_percent[
    #     'inst_cnt_installed']  # inst_cate_percent
    # df_result = pd.merge(df_result, df_percent, how='left',
    #                      on=['userID', 'appCategory'])

    # df_result['inst_cate_percent'].fillna(0.0, inplace=True)  # 同类应用比例
    # df_result['inst_cnt_installed'].fillna(0, inplace=True)

    # df_installed['count'] = np.ones(df_installed.shape[0])
    # df_group_exist = df_installed.groupby(['userID', 'appID']).count().rename(
    #     columns={'count': 'inst_is_installed'}).reset_index()
    # df_result = pd.merge(df_result, df_group_exist,
    #                      how='left', on=['userID', 'appID'])
    # df_result['inst_is_installed'].fillna(0, inplace=True)
    # del df_installed['count']

    # df_group_app = df_installed.groupby('appID').count().rename(
    #     columns={'userID': 'inst_app_installed'}).reset_index()
    # df_result = pd.merge(df_result, df_group_app, on='appID', how='left')
    # df_result['inst_app_installed'].fillna(0, inplace=True)

    # df_result['inst_cnt_installed'] = df_result['inst_cnt_installed'].fillna(
    #     0).astype('int64')  # 用戶已經安裝的app個數
    # df_result['inst_is_installed'] = df_result['inst_is_installed'].fillna(
    #     0).astype('int64')  # 該app被安裝的次数
    # df_result['inst_cnt_appcate'] = df_result['inst_cnt_appcate'].fillna(
    #     0).astype('int64')  # 同类应用个数
    # df_result['inst_app_installed'] = df_result['inst_app_installed'].fillna(
    #     0).astype('int64')  # 该app被安装的次数

    # 安裝流水中是否存在该应用
    # print '== get user actions feature =='
    # df_actions = read_as_pandas(FILE_USER_APP_ACTIONS)

    # df_result['index'] = range(df_result.shape[0])
    # df_merge = pd.merge(df_actions, df_app_category, how='left', on='appID').rename(
    #     columns={'appID': 'action_appID', 'appCategory': 'action_appCategory'})
    # df_merged = pd.merge(df_result, df_merge, on=['userID'], how='left')
    # df_merged['action_installed'] = (df_merged['clickTime'] > df_merged['installTime']) \
    #     & (df_merged['action_appID'] == df_merged['appID'])

    # df_merged['action_cate'] = (df_merged['clickTime'] > df_merged['installTime']) \
    #     & (df_merged['appCategory'] == df_merged['action_appCategory'])

    # df_merged['action_cate_recent'] = (df_merged['clickTime'] > df_merged['installTime']) \
    #     & (df_merged['clickTime'] - df_merged['installTime'] < 10000) \
    #     & (df_merged['appCategory'] == df_merged['action_appCategory'])  # 最近两天同类

    # df_merged['action_installed'] = df_merged['action_installed'].astype(
    #     'int64')
    # df_merged['action_cate'] = df_merged['action_cate'].astype('int64')
    # df_merged['action_cate_recent'] = df_merged['action_cate_recent'].astype(
    #     'int64')

    # df_sum = pd.DataFrame(df_merged.loc[:, ['index', 'action_installed', 'action_cate', 'action_cate_recent']]) \
    #     .groupby(['index']).sum().reset_index()

    # df_result['action_installed'] = df_sum['action_installed'].fillna(
    #     0).astype('int64')  # 用户安装该app的次数
    # df_result['action_cate'] = df_sum['action_cate'].fillna(
    #     0).astype('int64')  # 用户安装该类别的次数
    # df_result['action_cate_recent'] = df_sum['action_cate_recent'].fillna(
    #     0).astype('int64')  # 用户最近两天安装该类别app的次数

    # 修正已安装数据
    # df_result['tt_is_installed'] = (df_result['inst_is_installed'] > 0) \
    #     | (df_result['action_installed'] > 0)  # clickTime之前是否已经安装过

    # df_result['tt_is_installed'] = df_result['tt_is_installed'].astype('int64')
    # df_result['tt_cnt_appcate'] = df_result['action_cate'] + \
    #     df_result['inst_cnt_appcate']  # clickTime之前该app同类应用安装次数
    # df_result['tt_cnt_appcate'] = df_result['tt_cnt_appcate'].astype('int64')

    # context
    print '== make context feature =='
    df_result['clickTime_day'] = pd.Series(
        df_result['clickTime'].astype(int) // 1000000).astype(int)
    df_result['clickTime_hour'] = pd.Series(
        df_result['clickTime'].astype(int) % 1000000 // 10000).astype(int)
    df_result['clickTime_minute'] = pd.Series(
        df_result['clickTime'].astype(int) % 10000 // 100).astype(int)
    df_result['clickTime_second'] = pd.Series(
        df_result['clickTime'].astype(int) % 100).astype(int)

    df_result['clickTime_week'] = pd.Series(
        df_result['clickTime_day'] % 7).astype(int)

    # remove unrelated
    to_drop += ['clickTime']

    if for_train:
        to_drop += ['conversionTime']

    df_result.drop(to_drop, axis=1, inplace=True)
    print df_result.columns
    print df_result.head(5)
    return df_result, not_ohe


def get_tf_feature(with_ohe=False, save=True, _shuffle=False):
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print '重新生成特徵'
        df_train, not_ohe = get_feature(True)
        df_test, not_ohe = get_feature(False)

        # df_train.fillna(0, inplace=True)
        # df_test.fillna(0, inplace=True)

        df_concate = pd.concat([df_train, df_test], axis=0)
        get_hist_feature(
            [
                'userID',
                'creativeID',
                'positionID',
                'adID',
                'camgaignID',
                'advertiserID',
                'appID',
                'sitesetID',
                'appCategory',
                #   'appPlatform',
                #   'education',
                #   'gender',
                #   'haveBaby',
                #   'marriageStatus',
                #   'positionType',
                #   'hometown_c',
                #   'hometown_p',
                #   'residence_c',
                #   'residence_p',
                #   'telecomsOperator',
                'connectionType',
                #   'clickTime_week',
            ],
            df_concat=df_concate)

        # cnt_creativeID_positionID
        # df_cnt_creativeID_positionID = get_cnt_creativeID_positionID()
        # df_concate = pd.merge(df_concate, df_cnt_creativeID_positionID, how='left', on=[
        #                       'creativeID', 'positionID'])
        for column in df_concate.columns:
            print column, df_concate[column].dtype, df_concate[column].unique().shape, \
                df_concate[column].min(), df_concate[column].max()

        # df_concate.instanceID.fillna(-1, inplace=True)
        # df_train = (df_concate.loc[df_concate.instanceID <= 0]).drop(
        #     ['instanceID'], axis=1).copy()  # 重新赋值

        # # if _shuffle:  # 不对
        # #     shuffle(df_train)

        # df_test = (df_concate.loc[df_concate.instanceID > 0]).copy()  # 重新赋值
        # # 根据instanceID来判断，并且排序 确保正确
        df_test.sort_values('instanceID', inplace=True)
        if save:
            # df_train.to_csv('train.csv', index=False)
            # df_test.to_csv('test.csv', index=False)
            df_train.to_hdf('train.hdf5', key='train.hdf5')
            df_test.to_hdf('test.hdf5', key='test.hdf5')
    else:
        print '從文件加載特徵'
        df_train, not_ohe, df_test = pd.read_csv(
            'train.csv'), None, pd.read_csv('test.csv')

        # if _shuffle:
        #     shuffle(df_train)

    return df_train, df_test

    # if with_ohe and not_ohe is not None:
    #     idx_to_ohe = [i for i, j in enumerate(columns) if j not in not_ohe]
    #     encoder = OneHotEncoder(categorical_features=idx_to_ohe)
    #     df_concate = pd.concat([df_train, df_test], axis=1)
    #     df_concate.fillna(-1)
    #     encoder.fit(df_concate.values)
    #
    #     train_x = encoder.transform(train_x)
    #     test_x = encoder.transform(test_x)
    # if not_ohe is None and with_ohe:
    #     print 'not ohe is None , not going to ohe'

    # print train_x.shape, type(train_x), train_y.shape, type(train_y)

    # if save:
    #     pickle.dump(train_x, open('train_x.pkl', 'wb'), 2)
    #     pickle.dump(train_y, open('train_y.pkl', 'wb'), 2)

    #     pickle.dump(test_x, open('test_x.pkl', 'wb'), 2)
    #     pickle.dump(inst_id, open('inst_id.pkl', 'wb'), 2)

    # return train_x, train_y, test_x, test_y, pre_x, inst_id


def to_ffm():
    pass


def load_feature(from_file=True, with_ohe=True):
    """
    从文件加载或者。。。
    """
    if from_file:
        filenames = ['train_x.pkl', 'train_y.pkl', 'test_x.pkl', 'inst_id.pkl']
        objs = [pickle.load(open(f, 'rb')) for f in filenames]
        return objs
    else:
        return get_tf_feature(with_ohe=with_ohe)


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
            train_x, test_x, train_y, test_y = train_test_split(
                x.drop(['label'], axis=1).values,
                x['label'].values,
                test_size=test_size,
                stratify=y)
        else:
            train_x, test_x, train_y, test_y = train_test_split(
                x.drop(['label'], axis=1).values,
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
    train_x = x.loc[x['clickTime_day'] <= 30 - test_day_size, :].copy()
    test_x = x.loc[x['clickTime_day'] > 30 - test_day_size, :].copy()
    return train_x, test_x


if __name__ == '__main__':
    # df = get_feature(False)
    # print df.head(5)
    # load_feature(from_file=True, with_ohe=False)
    get_tf_feature(with_ohe=False)
    # get_cnt_creativeID_positionID()
    # get_hist_feature(['creativeID'], with_count=True)
