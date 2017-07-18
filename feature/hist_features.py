# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime


def get_hist_feature(vn,
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
    print '>>>> get hist of', vn
    df_concat['cvt_' + vn] = np.zeros(df_concat.shape[0])
    if with_count:
        df_concat['cnt_' + vn] = np.zeros(df_concat.shape[0])

    if with_pre_day_cnt:
        df_concat['cnt_' + vn] = np.zeros(df_concat.shape[0])

    # 第十七天使用当天的妆化率
    _pre_cnt = None
    _pre_sum = None
    for i in range(17, 32):
        print 'get hist of {}, day {}'.format(vn, i)
        df_concat['key'] = df_concat[vn].astype('category').values.codes
        if i > 17:
            df_grp_ = df_concat.loc[df_concat['clickTime_day'] < i,
                                    ['label', 'key']]

        else:
            df_grp_ = df_concat.loc[df_concat['clickTime_day'] == i,
                                    ['label', 'key']]
        # 当前天
        pre_grp_ = df_concat.loc[df_concat['clickTime_day'] == i,
                                 ['label', 'key']]
        pre_grp = pre_grp_
        pre_cnt = pre_grp.groupby('key').aggregate(np.size)
        pre_sum = pre_grp.groupby('key').aggregate(np.sum)

        # 历史
        df_grp = df_grp_
        cnt = df_grp.groupby('key').aggregate(np.size)
        sum = df_grp.groupby('key').aggregate(np.sum)
        v_codes = df_concat.loc[df_concat['clickTime_day'] == i, 'key'].values

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
                    df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cnt_' +
                                  vn] = _cnt.astype('float64')

        elif _pre_cnt is None and _pre_sum is None:  # 第17天的前一天 设置为第17天本身
            df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cvt_' +
                          vn] = df_concat.loc[df_concat['clickTime_day'] == i,
                                              'cvt_' + vn]
            if with_pre_day_cnt:
                df_concat.loc[df_concat['clickTime_day'] == i, 'pre_cnt_' +
                              vn] = df_concat.loc[df_concat['clickTime_day'] ==
                                                  i, 'cnt_' + vn]

        _pre_cnt = pre_cnt
        _pre_sum = pre_sum

    df_concat.drop(['key'], axis=1, inplace=True)
    # start = datetime.now()
    # df_concat.to_hdf('df_hist_feature_{}.hdf'.format(vn))
    # print datetime.now() - start
    start = datetime.now()
    df_concat.to_csv('df_hist_feature_{}.csv'.format(vn))
    print datetime.now() - start


hist_list = [
    # 'positionID',
    'userID',
    'creativeID',
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
    'clickTime_week',
]


def gen_by_day():
    df_train = pd.read_csv('df_basic_train.csv')
    df_test = pd.read_csv('df_basic_test.csv')

    concat = pd.concat([df_train, df_test]).reset_index(drop=True)
    for vn in hist_list:
        print vn
        get_hist_feature(
            vn, concat.loc[:, ['label', 'instanceID', 'clickTime_day', vn]])


def merge_days():
    df_result = None
    for c in hist_list:
        print c
        df = pd.read_csv("df_hist_feature_{}.csv".format(c))
        print 'read complete...'
        if df_result is None:
            df.drop([c], axis=1, inplace=True)
            df_result = df

        else:
            s = pd.Series(df['label'] != df_result['label']).astype(int).sum()
            assert s == 0
            df.drop(['label', 'instanceID', 'clickTime_day', c],
                    axis=1, inplace=True)
            df_result = pd.concat([df_result, df], axis=1)
            del df
            print df_result.summary()

    print df_result.shape
    print df_result.columns.values
    df_result.to_csv('df_hist_feature_all.csv', index=False, float_format='.7f')


def merge_chunk():
    dfs = [pd.read_csv("df_hist_feature_{}.csv".format(f),
                       iterator=True) for f in hist_list]
    loop = True
    chunk_size = 100000
    idx = 0
    while loop:
        idx += 1
        print idx,
        try:
            chunks = [m.get_chunk(chunk_size) for m in dfs]
            for df in chunks[1:]:
                df.drop(['label', 'instanceID', 'clickTime_day'],
                        axis=1, inplace=True)

            df_result = pd.concat(chunks, axis=1)
            print df_result.shape
            df_result.drop(hist_list, axis=1, inplace=True)
            df_result.to_csv('df_hist_feature_all.csv', mode='a+', index=False)

        except StopIteration:
            loop = False
            print "iteration stops"


if __name__ == '__main__':
    # merge_days()
    # gen_by_day()
    merge_chunk()
