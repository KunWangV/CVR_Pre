# coding: utf-8

import pandas as pd
import numpy as np
from feature.data import *

# 用户已安装列表是否存在该应用、同类应用的数量、所占比例、该用户已经安装app的数量
print
'== get install feature =='

df_installed = read_as_pandas(FILE_USER_INSTALLEDAPPS)
df_app_category = read_as_pandas(FILE_APP_CATEGORIES)

df_group_cnt = df_installed.groupby('userID').count().rename(
    columns={'appID': 'inst_cnt_installed'}).reset_index()
df_installed_cate = pd.merge(
    df_installed, df_app_category, how='left', on='appID')
df_group_cnt_user_appcate = df_installed_cate.groupby(
    ['userID', 'appCategory']).count().reset_index().rename(
    columns={'appID': 'inst_cnt_appcate'})
df_percent = pd.merge(
    df_group_cnt_user_appcate, df_group_cnt, how='left',
    on='userID')  # userID, appCategory, inst_cnt_installed, inst_cnt_appcate
df_percent['inst_cate_percent'] = df_percent['inst_cnt_appcate'].astype(
    float) / df_percent['inst_cnt_installed']  # inst_cate_percent

df_installed['count'] = np.ones(df_installed.shape[0])
df_group_exist = df_installed.groupby(['userID', 'appID']).count().rename(
    columns={'count': 'inst_is_installed'}).reset_index()

df_group_app = df_installed.groupby('appID').count().rename(
    columns={'userID': 'inst_app_installed'}).reset_index()

df_train = pd.read_csv('df_basic_train.csv')
df_test = pd.read_csv('df_basic_test.csv')
df_result = pd.concate(df_train, df_test)

df_result = pd.merge(
    df_result, df_percent, how='left', on=['userID', 'appCategory'])
df_result['inst_cate_percent'].fillna(0.0, inplace=True)  # 同类应用比例
df_result['inst_cnt_installed'].fillna(0, inplace=True)

df_result = pd.merge(
    df_result, df_group_exist, how='left', on=['userID', 'appID'])
df_result['inst_is_installed'].fillna(0, inplace=True)
del df_installed['count']

df_result = pd.merge(df_result, df_group_app, on='appID', how='left')
df_result['inst_app_installed'].fillna(0, inplace=True)

df_result['inst_cnt_installed'] = df_result['inst_cnt_installed'].fillna(
    0).astype('int64')  # 用戶已經安裝的app個數
df_result['inst_is_installed'] = df_result['inst_is_installed'].fillna(
    0).astype('int64')  # 該app被安裝的次数
df_result['inst_cnt_appcate'] = df_result['inst_cnt_appcate'].fillna(0).astype(
    'int64')  # 同类应用个数
df_result['inst_app_installed'] = df_result['inst_app_installed'].fillna(
    0).astype('int64')  # 该app被安装的次数
