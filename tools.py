# coding: utf-8

import pandas as pd

from utils import gen_column_info_list, map_by_chunk, save_pandas, read_as_pandas
import config

#
# map_by_chunk('../hist_features/df_hist_feature_all.csv',
#              read_func=lambda x: pd.read_csv(x, iterator=True),
#              map_func=lambda df: df.drop(df.columns[df.columns.str.startswith('Unnamed')], axis=1),
#              save_func=lambda df: df.to_csv('../hist_features/df_hist_feature_all_clean.csv', mode='a+',
#                                             float_format='.7f'))
#
#

# """
# 用来过滤数据
# """
# s = 17
# e = 30
# map_by_chunk(
#     'train.hdf5',
#     read_func=lambda filename: read_as_pandas(filename, by_chunk=True),
#     map_func=lambda df: df.loc[(df['clickTime_day'] >= s) & (df['clickTime_day'] <= e), :],
#     save_func=lambda df: save_pandas(df, 'result.hdf5', append=True),
# )


"""
增加一列
"""
s = 17
e = 30

def map_func1(df):
    df['clickTime_day'] = df['clickTime'].astype(int) // 1000000
    return df

map_by_chunk(
    '../train.csv',
    read_func=lambda filename: read_as_pandas(filename, by_chunk=True),
    map_func=map_func1,
    save_func=lambda df: save_pandas(df, '../train_dc.csv', append=True, index=False),
)

# dataframe summary
