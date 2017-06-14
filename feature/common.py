# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd


class ColumnInfo(object):
    """
    每列信息类
    """

    def __init__(self, name, type, unique_size=None, dtype='int64'):
        """
        :param name: 列名
        :param type:  category or real
        :param unique_size: 最大值
        :param dtype: 数据类型
        """
        self.name = name
        self.type = type
        self.unique_size = unique_size
        self.dtype = dtype

    def __str__(self):
        return 'name: {}, type: {}, max value: {}, dtype: {}'.format(
            self.name, self.type, self.unique_size, self.dtype)


def gen_column_list(df,
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
    infos = []
    for c in columns:
        print(c)
        if c in cate_feats and c not in drop_feats:
            df[c] = df[c].astype('int64')
            info = ColumnInfo(
                name=c,
                type='category',
                unique_size=df[c].max(),
                dtype='int64')

            infos.append(info)

        elif c in real_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='real',
                dtype=str(df[c].dtype),
                unique_size=None, )
            infos.append(info)

        else:
            print('unknow column')

    if save:
        pickle.dump(infos, open(save_name, 'wb'))

    return infos


def columns_from_column_infos(infos):
    column_list = []
    for info in infos:
        column_list.append(info.name)

    return column_list


def load_pickle(filename):
    return pickle.load(open(filename, 'r'))


def save_pickle(obj, filename):
    pickle.dump(obj, open(filename, 'w'))


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


def read_hdf(filename):
    return pd.read_hdf(filename)


def save_as_hdf(df, filename, key=None):
    if key == None:
        key = filename

    df.to_hdf(filename, key=key)


def read_as_pandas(filename):
    if filename.endswith('.hdf5') or filename.endswith('.hdf'):
        return pd.read_hdf(filename)

    else:
        return pd.read_csv(filenae)
        