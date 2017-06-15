# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
import os
import sys
import numpy as np


def save_as_hdf(df, filename, key=None):
    """
    保存模型
    :param df:
    :param filename:
    :param key:
    :return:
    """
    if key == None:
        key = filename

    df.to_hdf(filename, key=key)


def read_hdf(filename):
    """
    保存hdf
    :param filename:
    :return:
    """
    return pd.read_hdf(filename)


def read_as_pandas(filename, by_chunk=False):
    """
    读取文件
    :param filename:
    :return:
    """
    if filename.endswith('.hdf5') or filename.endswith('.hdf'):
        return pd.read_hdf(filename, iterator=by_chunk)

    else:
        return pd.read_csv(filename, iterator=by_chunk)


def save_pandas(df, filename, key=None, append=False):
    """
    保存 支持分chunk
    :param df:
    :param filename:
    :param key:
    :param append:
    :return:
    """
    if key is None:
        key = filename

    if filename.endswith('.hdf5') or filename.endswith('.hdf'):
        if append:
            df.to_hdf(filename, key=key, mode='a+', append=True)

        else:
            df.to_hdf(filename, key=key)
    else:
        if append:
            df.to_csv(filename, mode='a+')
        else:
            df.to_csv(filename, mode='a+')


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def save_pickle(obj, filename):
    if os.path.exists(filename):
        print("warning: 文件已经存在，即将被覆盖")

    pickle.dump(obj, open(filename, 'wb'))


class ColumnInfo(object):
    """
    每列信息类
    """

    def __init__(self, name, type, max_val=None, min_val=None, total=None, unique_size=None, dtype='int64'):
        """
        :param name: 列名
        :param type:  category or real
        :param max_val: 最大值
        :param min_val: 最小值
        :param unique_size: 最大值
        :param dtype: 数据类型
        """
        self.name = name
        self.type = type
        self.max_val = max_val
        self.min_val = min_val
        self.unique_size = unique_size
        self.total = total
        self.dtype = dtype

    def __str__(self):
        return 'name: {}, type: {}, max value: {}, min value {}, unique value: {}, dtype: {}'.format(
            self.name, self.type, self.max_val, self.min_val, self.unique_size, self.dtype)


def gen_column_info_list(df,
                         cate_feats,
                         real_feats,
                         drop_feats,
                         save=True,
                         save_name='column_list.pkl'):
    """
    构造统计信息
    :param df:
    :param cate_feats:
    :param real_feats:
    :param drop_feats:
    :param save:
    :param save_name:
    :return:
    """
    columns = df.columns.values
    print(columns)
    infos = []
    for c in columns:
        print(c)
        if c in cate_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='category',
                max_val=df[c].max(),
                min_val=df[c].min(),
                unique_size=df[c].unique(),
                dtype='int64',
            )

            infos.append(info)
            print(str(info))

        elif c in real_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='real',
                dtype=str(df[c].dtype),
                unique_size=None,
                max_val=df[c].max(),
                min_val=df[c].min(),
            )
            infos.append(info)
            print(str(info))
        else:
            print('unknow column....')

    if save:
        save_pickle(infos, save_name)

    return infos


def get_columns_from_column_infos(infos):
    """
    获取列名
    :param infos:
    :return:
    """
    column_list = []
    for info in infos:
        column_list.append(info.name)

    return column_list


def data_transform(df,
                   real_feats,
                   cate_feats,
                   drop_feats,
                   to_cate=False,
                   to_cvt_type=False,
                   to_drop=False,
                   to_log_real=False,
                   to_fill_na=False,
                   log_threshold=2,
                   ):
    """

    :param df:
    :param real_feats:
    :param cate_feats:
    :param drop_feats:
    :param to_cate: values.codes
    :param to_cvt_type: 转换数据类型 cate => int, real=>float
    :param to_drop: 删除drop特征
    :param to_log_real: log2
    :param log_threshold
    :return:
    """
    columns = df.columns.values
    print(columns)

    if to_drop and len(drop_feats) > 0:
        df.drop(drop_feats, axis=1, inplace=True)

    int_max = 2 ** 31
    for c in columns:
        print(c)
        if c in cate_feats and c not in drop_feats:
            if to_cate:
                df[c] = df[c].astype('category', copy=False).values.codes

            if to_fill_na:
                df[c].fillna(0, inplace=True)

        elif c in real_feats and c not in drop_feats:
            if to_cvt_type:
                df[c] = df[c].astype('float32', copy=False)

            if to_log_real:
                df.loc[df[c] > log_threshold, c] = np.power(np.log(df.loc[df[c] > log_threshold, c].values), 2)

            if to_fill_na:
                df[c].fillna(df[c].mean(), inplace=True)

        else:
            print('unknow columns..... to drop')
            del df[c]


def map_by_chunk(filename, read_func, save_func, map_func, chunk_size=100000):
    """
    逐map修改
    :param filename:
    :param read_func: args-filename
    :param save_func: args-dataframe with return
    :param map_func: args-dataframe with return
    :param chunk_size:
    :return:
    """
    m = read_func(filename)
    loop = True
    idx = 0
    while loop:
        idx += 1
        print(
            idx, )
        try:
            chunk = m.get_chunk(chunk_size)
            print(
                chunk.shape, )
            if map_func is not None:
                chunk = map_func(chunk)

            print('after map', chunk.shape)
            save_func(chunk)
        except StopIteration:
            loop = False
            print("iteration stops")


def merge_by_chunk(
        filenames,
        read_func,
        save_func,
        map_func,
        chunk_size=100000, ):
    """
    把文件合并
    :param filenames: list filename
    :param read_func: args-filename return dataframe
    :param save_func: args-dataframe
    :param map_func: args-dataframe return dataframe
    :param chunk_size:
    :return:
    """
    dfs = [read_func(filename) for filename in filenames]
    loop = True
    idx = 0
    while loop:
        idx += 1
        print(
            idx, )
        try:
            chunks = [m.get_chunk(chunk_size) for m in dfs]
            for i, df in enumerate(chunks):
                chunks[i] = map_func(i, df)

            df_result = pd.concat(chunks, axis=1)
            print(df_result.shape)
            save_func(df_result)
        except StopIteration:
            loop = False
            print("iteration stops")


class PandasChunkReader(object):
    """
    支持pandas文件循环读取
    """

    def __init__(self, filename, chunk_size=100000, loop=False):
        """

        :param filename:
        :param chunk_size:
        :param loop: 一直循环，重复读取
        """
        self.df = read_as_pandas(filename, iterator=True)
        self.filename = filename
        self.chunk_size = chunk_size
        self.loop = loop
        self.epoch = 0

    def reset_df(self):
        self.df = read_as_pandas(self.filename, iterator=True)

    def next(self):
        try:
            df_chunk = self.df.get_chunk(self.chunk_size)
        except StopIteration:
            if self.loop:
                self.epoch += 1
                self.reset_df()
                self.next()
            else:
                print("iterator stops")
