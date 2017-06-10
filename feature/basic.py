# coding: utf-8

import pandas as pd
import numpy as np
from data import *
from datetime import datetime


def make_basic(for_train):
    start = datetime.now()
    if for_train:
        df_file = read_as_pandas(FILE_TRAIN)
    else:
        df_file = read_as_pandas(FILE_TEST)

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
    df_user['hometown_p'] = np.round(
        df_user['hometown'].astype(int) / 100).astype(int)  # 取省份
    df_user['hometown_c'] = np.round(
        df_user['hometown'].astype(int) % 100).astype(int)  # 城市
    df_user['residence_p'] = np.round(
        df_user['residence'].astype(int) / 100).astype(int)  # 取省份
    df_user['residence_c'] = np.round(
        df_user['residence'].astype(int) % 100).astype(int)  # 城市
    df_result = pd.merge(df_result, df_user, how='left', on='userID')

    # position related
    print '== get position feature =='
    df_position = read_as_pandas(FILE_POSITION)
    df_result = pd.merge(df_result, df_position, how='left', on='positionID')

    # context
    print '== make context feature =='
    df_result['clickTime_day'] = np.floor(
        df_result['clickTime'].astype(int) / 1000000).astype(int)
    df_result['clickTime_hour'] = np.floor(
        df_result['clickTime'].astype(int) % 1000000 / 10000).astype(int)
    df_result['clickTime_minute'] = np.floor(
        df_result['clickTime'].astype(int) % 10000 / 100).astype(int)
    df_result['clickTime_seconds'] = np.floor(
        df_result['clickTime'].astype(int) % 100).astype(int)

    df_result['clickTime_week'] = df_result['clickTime_day'].astype(int) % 7

    # remove unrelated
    to_drop += [
        'clickTime',
        'hometown',
        'residence',
    ]

    if for_train:
        to_drop += ['conversionTime']

    df_result.drop(to_drop, axis=1, inplace=True)
    print df_result.columns
    print df_result.head(5)

    end = datetime.now()
    if for_train:
        df_result.to_csv('df_basic_train.csv', index=False)
    else:
        df_result.to_csv('df_basic_test.csv', index=False)
    print "start: {} end: {} used: {}".format(start, end, end - start)

    return df_result


def make_train_test(df):

    if df is None:
        df = pd.read_csv('df_basic_train.csv')

    print df['clickTime_day'].unique()

    df_test = df.loc[df['clickTime_day'] >= 29, :]

    df_train = df.loc[df['clickTime_day'] < 29, :]

    df_train_y = pd.DataFrame(df_train.loc[:, 'label']).copy()
    df_test_y = pd.DataFrame(df_test.loc[:, 'label']).copy()
    df_test_y.columns = ['label']
    df_train_y.columns = ['label']

    print df_train.shape, df_test.shape, df_train_y.shape, df_test_y.shape

    df_train_y.to_csv('df_trainy.csv', index=False)
    df_test_y.to_csv('df_testy.csv', index=False)

    # del df_train['label']
    # del df_test['label']

    # df_train.to_csv('df_trainx.csv', index=False)
    # df_test.to_csv('df_testx.csv', index=False)


if __name__ == '__main__':
    # result = make_basic(True)
    # make_basic(False)
    make_train_test(None)
