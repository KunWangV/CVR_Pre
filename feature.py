# coding: utf-8
# pylint: disable=C0103, C0111

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
from pyspark.ml.feature import OneHotEncoder as OHE
import pyspark.sql.functions as F
import numpy as np

from data import *
import pdb
import pickle

from tqdm import tqdm


def get_feature(for_train=True):
    """
    提取相关特征

    u'label':
    : id类
    u'instanceID'
    u'creativeID'
    u'userID'
    u'positionID',
    u'adID'
    u'camgaignID',
    u'advertiserID'
    u'appID'

    : category 类
    u'connectionType'
    u'telecomsOperator'
    u'appPlatform'
    u'appCategory'
    u'age',
    u'gender'
    u'education'
    u'marriageStatus'
    u'haveBaby'
    u'hometown_p',
    u'hometown_c'
    u'residence_p'
    u'residence_c'
    u'sitesetID',
    u'positionType'

    : 连续值类
    u'inst_cnt_appcate'
    u'inst_cnt_installed',
    u'inst_cate_percent'
    u'inst_is_installed'
    u'inst_app_installed',
    u'action_installed'
    u'action_cate'
    u'action_cate_recent',
    u'tt_is_installed'
    u'tt_cnt_appcate'
    u'clickTime_day',
    u'clickTime_hour'
    u'clickTime_minute'
    """
    if for_train:
        df_file = read_as_pandas(FILE_TRAIN)

    else:
        df_file = read_as_pandas(FILE_TEST)

    not_ohe = []
    to_drop = []

    # creative realated
    df_ad = read_as_pandas(FILE_AD)  # creativeID,adID,camgaignID,advertiserID,appID,appPlatform
    df_app_category = read_as_pandas(FILE_APP_CATEGORIES)
    df_result = pd.merge(df_file, df_ad, how='left', on='creativeID')
    df_result = pd.merge(df_result, df_app_category, how='left', on='appID')

    # user realated
    df_user = read_as_pandas(FILE_USER)
    df_user['age'] = np.round(df_user['age'] / 10).astype(int)  # 年龄段
    df_user['hometown_p'] = np.round(df_user['hometown'].astype(int) / 100).astype(int)  # 取省份
    df_user['hometown_c'] = np.round(df_user['hometown'].astype(int) % 100).astype(int)  # 城市
    df_user['residence_p'] = np.round(df_user['residence'].astype(int) / 100).astype(int)  # 取省份
    df_user['residence_c'] = np.round(df_user['residence'].astype(int) % 100).astype(int)  # 城市
    df_result = pd.merge(df_result, df_user, how='left', on='userID')

    to_drop += [
        'hometown',
        'residence',
    ]

    # position related
    df_position = read_as_pandas(FILE_POSITION)
    df_result = pd.merge(df_result, df_position, how='left', on='positionID')

    ## installed app related
    # 用户已安装列表是否存在该应用、同类应用的数量、所占比例、该用户已经安装app的数量
    df_installed = read_as_pandas(FILE_USER_INSTALLEDAPPS)
    df_group_cnt = df_installed.groupby('userID').count().rename(
        columns={'appID': 'inst_cnt_installed'}).reset_index()
    df_installed_cate = pd.merge(df_installed, df_app_category, how='left', on='appID')
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
    df_result = pd.merge(df_result, df_group_exist, how='left', on=['userID', 'appID'])
    df_result['inst_is_installed'].fillna(0, inplace=True)
    del df_installed['count']

    df_group_app = df_installed.groupby('appID').count().rename(
        columns={'userID': 'inst_app_installed'}).reset_index()
    df_result = pd.merge(df_result, df_group_app, on='appID', how='left')
    df_result['inst_app_installed'].fillna(0, inplace=True)

    df_result['inst_cnt_installed'] = df_result['inst_cnt_installed'].astype(int)  # 用戶已經安裝的app個數
    df_result['inst_is_installed'] = df_result['inst_is_installed'].astype(int)  # 該app被安裝的次数
    df_result['inst_cnt_appcate'] = df_result['inst_cnt_appcate'].fillna(0).astype(int)  # 同类应用个数
    df_result['inst_app_installed'] = df_result['inst_app_installed'].astype(int)  # 该app被安装的次数

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
    df_merged['action_cate_recent'] = df_merged['action_cate_recent'].astype(int)
    df_sum = pd.DataFrame(df_merged.loc[:, ['index', 'action_installed', 'action_cate', 'action_cate_recent']]) \
        .groupby(['index']).sum().reset_index()

    df_result['action_installed'] = df_sum['action_installed'].fillna(0)  # 用户安装该app的次数
    df_result['action_cate'] = df_sum['action_cate'].fillna(0)  # 用户安装该类别的次数
    df_result['action_cate_recent'] = df_sum['action_cate_recent'].fillna(0)  # 用户最近两天安装该类别app的次数

    # 修正已安装数据
    df_result['tt_is_installed'] = (df_result['inst_is_installed'] > 0) \
                                   & (df_result['action_installed'] > 0)  # clickTime之前是否已经安装过

    df_result['tt_is_installed'] = df_result['tt_is_installed'].astype(int)
    df_result['tt_cnt_appcate'] = df_result['action_cate'] + df_result['inst_cnt_appcate']  # clickTime之前该app同类应用安装次数

    # context
    df_result['clickTime_day'] = df_result['clickTime'].astype(str).str.slice(0, 2)
    df_result['clickTime_hour'] = df_result['clickTime'].astype(str).str.slice(2, 4)
    df_result['clickTime_minute'] = df_result['clickTime'].astype(str).str.slice(4, 6)

    df_result['clickTime_day'] = df_result['clickTime_day'].astype(int)
    df_result['clickTime_hour'] = df_result['clickTime_hour'].astype(int)
    df_result['clickTime_minute'] = df_result['clickTime_minute'].astype(int)
    df_result['clickTime_week'] = df_result['clickTime'] / 10000 % 7

    # remove unrelated
    to_drop += ['clickTime', 'index']

    if for_train:
        to_drop += ['conversionTime']

    df_result.drop(to_drop, axis=1, inplace=True)
    print df_result.columns
    print df_result.head(5)
    return df_result, not_ohe


def get_tf_feature(with_ohe=True, save=True):
    df_train, not_ohe = get_feature(True)
    shuffle(df_train)
    train_y = np.round(df_train['label']).astype(int).values
    df_train.drop('label', 1, inplace=True)
    df_train.fillna(0, inplace=True)
    train_x = df_train.values
    columns = df_train.columns

    df_test, not_ohe = get_feature(False)
    df_test.fillna(0, inplace=True)
    inst_id = df_test['instanceID'].values
    df_test.drop(['label', 'instanceID'], axis=1, inplace=True)
    test_x = df_test.values

    if with_ohe:
        idx_to_ohe = [i for i, j in enumerate(columns) if j not in not_ohe]
        encoder = OneHotEncoder(categorical_features=idx_to_ohe)
        df_concate = pd.concat([df_train, df_test], axis=1)
        df_concate.fillna(-1)
        encoder.fit(df_concate.values)

        train_x = encoder.transform(train_x)
        test_x = encoder.transform(test_x)

    print train_x.shape, type(train_x), train_y.shape, type(train_y)

    if save:
        pickle.dump(train_x, open('train_x.pkl', 'wb'), 2)
        pickle.dump(train_y, open('train_y.pkl', 'wb'), 2)

        pickle.dump(test_x, open('test_x.pkl', 'wb'), 2)
        pickle.dump(inst_id, open('inst_id.pkl', 'wb'), 2)

    return train_x, train_y, test_x, inst_id


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


def split_train_test(x, y, test_size=0.2, stratify=True):
    """
    分割数据
    :param x:
    :param y:
    :param test_size:
    :param stratify: 考虑不平衡问题
    :return:
    """
    if stratify:
        return train_test_split(x, y, test_size=test_size)
    else:
        return train_test_split(x, y, test_size=test_size, stratify=y)


if __name__ == '__main__':
    # df = get_feature(False)
    # print df.head(5)
    get_tf_feature(with_ohe=False)
