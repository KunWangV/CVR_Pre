# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
import os
import sys
import numpy as np
import gc
from argparse import ArgumentParser


def ensure_exits(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_spark_sesssion():
    from pyspark.sql import SparkSession
    sess = SparkSession.builder.appName('tencent') \
        .config('spark.executor.memory', '16000m') \
        .config('spark.driver.memory', '16000m') \
        .master('local[4]') \
        .getOrCreate()

    return sess


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


def read_as_pandas(filename, by_chunk=False, chunk_size=100000):
    """
    读取文件
    :param filename:
    :return:
    """

    if filename.endswith('.hdf5') or filename.endswith('.hdf'):
        return pd.read_hdf(filename, iterator=by_chunk, chunksize=chunk_size)

    else:
        return pd.read_csv(filename, iterator=by_chunk, chunksize=chunk_size)


def save_pandas(df, filename, key=None, append=False, index=True):
    """
    保存 支持分chunk
    :param df:
    :param filename:
    :param key:
    :param append:
    :return:
    """

    if df.shape[0] == 0:
        return

    if key is None:
        key = filename

    if filename.endswith('.hdf5') or filename.endswith('.hdf'):
        if append:
            df.to_hdf(filename, key=key, mode='a', append=True)

        else:
            df.to_hdf(filename, key=key)
    else:
        if append:
            df.to_csv(filename, mode='a+', index=index)
        else:
            df.to_csv(filename, index=index)


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

    def __init__(self,
                 name,
                 type,
                 max_val=None,
                 min_val=None,
                 total=None,
                 unique_size=None,
                 dtype='int64'):
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
            self.name, self.type, self.max_val, self.min_val, self.unique_size,
            self.dtype)


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
                dtype='int64', )

            infos.append(info)
            print(str(info))

        elif c in real_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='real',
                dtype=str(df[c].dtype),
                unique_size=None,
                max_val=df[c].max(),
                min_val=df[c].min(), )
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


def data_transform(
        df,
        real_feats,
        cate_feats,
        drop_feats,
        to_cate=False,
        to_cvt_type=False,
        to_drop=False,
        to_log_real=False,
        to_fill_na=False,
        log_threshold=2, ):
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

    int_max = 2**31
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
                df.loc[df[c] > log_threshold, c] = np.power(
                    np.log(df.loc[df[c] > log_threshold, c].values), 2)

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
    idx = 0
    if not isinstance(m, pd.io.pytables.TableIterator):
        m = iter(m)

    for chunk in m:
        idx += 1
        print(idx, chunk.shape)
        if map_func is not None:
            chunk = map_func(chunk)

        print('after map', chunk.shape)
        save_func(chunk)
        del chunk
        gc.collect()


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
        print(idx)
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
        self.df = read_as_pandas(filename, iterator=True, chunk_size=self.chunk_size)
        self.it = iter(self.df)
        self.filename = filename
        self.chunk_size = chunk_size
        self.loop = loop
        self.epoch = 0

    def reset_df(self):
        self.df = read_as_pandas(self.filename, iterator=True, chunk_size=self.chunk_size)
        self.it = iter(self.df)

    def next(self):
        try:
            df_chunk = self.it.next()
            return df_chunk
        except StopIteration:
            if self.loop:
                self.epoch += 1
                self.reset_df()
                self.next()
            else:
                print("iterator stops")



def df_summary(df, outfile=None):
    """
    df or str
    :param filename:
    :return:
    """
    shape = None
    if isinstance(df, str):
        shape = read_as_pandas(df, by_chunk=False, chunk_size=None).shape
    else:
        shape = df.shape

    if outfile is None:
        outfile = 'df_summary_{}.pkl'.format(os.path.splitext(os.path.basename(df))[0])

    ensure_exits(os.path.dirname(outfile))
    save_pickle(np.asarray(shape), outfile)


def df_summary_by_chunk(filename, outfile, chunk_size=100000):
    """
    读取 把行信息设置为
    :param filename:
    :param chunk_size:
    :return:
    """
    global shape
    shape = None

    if outfile is None:
        outfile = 'df_summary_{}.pkl'.format(os.path.splitext(os.path.basename(filename))[0])

    def append(df):
        nshape = np.asarray(df.shape)
        global shape
        if shape is None:
            shape = nshape

        else:
            shape[0] += nshape[0]

        return df

    map_by_chunk(
        filename,
        map_func=lambda df: append(df),
        read_func=lambda fname: read_as_pandas(fname, by_chunk=True, chunk_size=chunk_size),
        save_func=lambda df: df,
    )

    print(shape, type(shape))

    ensure_exits(os.path.dirname(filename))
    save_pickle(shape, 'df_summary_{}.pkl'.format(os.path.splitext(os.path.basename(filename))[0]))


def main(args):
    if args.by_chunk:
        df_summary_by_chunk(args.filename, args.outfile)

    else:
        df_summary(args.filename, args.outfile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--by-chunk', type=bool, default=False)
    parser.add_argument('--outfile', type=str)

    args = parser.parse_args()
    main(args)
