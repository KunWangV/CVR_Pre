# coding: utf-8
# pylint: disable=C0103, C0111

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle

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
    """
    if for_train:
        df_file = read_as_pandas(FILE_TRAIN)

    else:
        df_file = read_as_pandas(FILE_TEST)

    not_ohe = []
    to_drop = []

    # creative realated
    df_ad = read_as_pandas(FILE_AD)
    df_ad = df_ad[['creativeID', 'appID', 'appPlatform']]
    df_app_category = read_as_pandas(FILE_APP_CATEGORIES)
    df_result = pd.merge(df_file, df_ad, how='left', on='creativeID')
    df_result = pd.merge(df_result, df_app_category, how='left', on='appID')

    # user realated
    df_user = read_as_pandas(FILE_USER)
    df_user['age'] = np.round(df_user['age'] / 10).astype(int)  # 年龄段
    df_user['hometown_p'] = np.round(df_user['hometown'] / 100).astype(
        int)  # 取省份
    df_user['hometown_c'] = np.round(df_user['hometown'] % 100).astype(
        int)  # 城市
    df_user['residence_p'] = np.round(df_user['residence'] / 100).astype(
        int)  # 取省份
    df_user['residence_c'] = np.round(df_user['residence'] % 100).astype(
        int)  # 城市
    df_result = pd.merge(df_result, df_user, how='left', on='userID')

    to_drop += [
        'hometown',
        'residence',
    ]

    # position related
    df_position = read_as_pandas(FILE_POSITION)
    df_result = pd.merge(df_result, df_position, how='left', on='positionID')
    # installed app related
    # 已安装列表是否存在该应用、同类应用的数量

    # 最近是否安装了该应用或者同类应用

    # 该用户已经安装app的数量

    # 该app被安装的数量

    # context
    df_result['clickTime_day'] = df_result['clickTime'].astype(str).str.slice(
        0, 2)
    df_result['clickTime_hour'] = df_result['clickTime'].astype(str).str.slice(
        2, 4)
    df_result['clickTime_minute'] = df_result['clickTime'].astype(
        str).str.slice(4, 6)

    # remove unrelated
    to_drop += ['clickTime', 'creativeID', 'userID', 'positionID', 'appID']

    if for_train:
        to_drop += ['conversionTime']

    df_result.drop(to_drop, axis=1, inplace=True)
    return df_result, not_ohe


def get_tf_feature(gen_ffm=False):
    df_train, not_ohe = get_feature(True)
    shuffle(df_train)
    train_y = np.round(df_train['label']).astype(int).values
    df_train.drop('label', 1, inplace=True)
    print df_train.columns
    df_train.fillna(0, inplace=True)
    train_x = df_train.values
    columns = df_train.columns

    df_test, not_ohe = get_feature(False)
    df_test.fillna(0, inplace=True)
    inst_id = df_test['instanceID'].values
    df_test.drop(['label', 'instanceID'], axis=1, inplace=True)
    test_x = df_test.values

    if gen_ffm:
        df_concate = pd.concat([df_train, df_test])
        list_count = [(c, df_concate[c].unique().shape[0])
                      for c in df_concate.columns]
        dict_column2field = {
            u'connectionType': 0,
            u'telecomsOperator': 0,
            u'clickTime_day': 0,
            u'clickTime_minute': 0,
            u'clickTime_hour': 0,
            u'appPlatform': 1,
            u'appCategory': 1,
            u'age': 2,
            u'gender': 2,
            u'education': 2,
            u'marriageStatus': 2,
            u'haveBaby': 2,
            u'hometown_p': 2,
            u'hometown_c': 2,
            u'residence_p': 2,
            u'residence_c': 2,
            u'sitesetID': 3,
            u'positionType': 3,
        }

    idx_to_ohe = [i for i, j in enumerate(columns) if j not in not_ohe]
    encoder = OneHotEncoder(categorical_features=idx_to_ohe)
    if df_concate is not None:
        df_concate.fillna(0, inplace=True)
        np_concate = df_concate.values
    else:
        np_concate = np.concatenate([train_x, test_x], axis=0)

    print np_concate.shape
    encoder.fit(np_concate)

    train_x = encoder.transform(train_x)
    print train_x.shape, type(train_x), train_y.shape, type(train_y)
    pickle.dump(train_x, open('train_x.pkl', 'wb'), 2)
    pickle.dump(train_y, open('train_y.pkl', 'wb'), 2)

    test_x = encoder.transform(test_x)
    pickle.dump(test_x, open('test_x.pkl', 'wb'), 2)
    pickle.dump(inst_id, open('inst_id.pkl', 'wb'), 2)

    if gen_ffm:
        columns_labels = []
        for label, count in list_count:
            columns_labels += [label] * count

        columns_labels = np.asarray(columns_labels)

        def to_fm(filename, data, pre=False):
            with open(filename, 'w') as f:
                for i in tqdm(range(data.shape[0])):
                    row_indice = data.getrow(i).nonzero()
                    row_values = data[row_indice]
                    row_field = np.asarray([
                        dict_column2field[c]
                        for c in columns_labels[row_indice[1]]
                    ])
                    # print row_indice[1].shape, row_values.shape, row_field.shape
                    line = [
                        '{}:{}:{}'.format(row_field[i], row_indice[1][i],
                                          1)
                        for i in range(len(row_field))
                    ]
                    if not pre:
                        line = [str(train_y[i])] + line
                    else:
                        line = [str(-1)] + line
                    f.write(' '.join(line) + "\n")

                f.write('\n')

        print train_x.shape, test_x.shape
        to_fm('train_x.ffm', train_x)
        to_fm('test_x.ffm', test_x, True)

    return train_x, train_y, test_x, inst_id


def to_ffm():
    pass


def load_feature(from_file=True):
    """
    从文件加载或者。。。
    """
    if from_file:
        filenames = ['train_x.pkl', 'train_y.pkl', 'test_x.pkl', 'inst_id.pkl']
        objs = [pickle.load(open(f, 'rb')) for f in filenames]
        return objs
    else:
        return get_tf_feature()


if __name__ == '__main__':
    # df = get_feature(False)
    # print df.head(5)
    get_tf_feature(gen_ffm=True)
