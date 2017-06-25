# coding: utf-8
from __future__ import print_function
from utils import *
import os
from config import *


def df_infos_summary(filename, save=True, save_name="column_summary.pkl"):
    """
    获取dataframe 信息
    :param filename:
    :param save:
    :param save_name:
    :return:
    """
    print("df_infos")
    sess = get_spark_sesssion()
    dataframe = sess.read.load(filename, format=os.path.splitext(filename)[
                                                    1][1:], header=True, inferSchema=True)

    mins = dataframe.groupby().min().collect()[0]
    maxs = dataframe.groupby().max().collect()[0]

    min_maxs = zip(mins, maxs)

    uniques = []
    for h in dataframe.columns:
        uniques.append(dataframe.select(h).distinct().count())

    columns = dataframe.columns
    infos = []
    for i, (min_val, max_val) in enumerate(min_maxs):
        c = columns[i]
        print(c)
        if c in cate_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='category',
                max_val=max_val,
                min_val=min_val,
                unique_size=uniques[i],
                dtype='int64',
            )

            infos.append(info)
            print(str(info))

        elif c in real_feats and c not in drop_feats:
            info = ColumnInfo(
                name=c,
                type='real',
                unique_size=None,
                max_val=max_val,
                min_val=min_val,
            )
            infos.append(info)
            print(str(info))
        else:
            print('unknow column....', c)

    save_pickle(infos, save_name)


def normalize(data_file, train_file, test_file):
    """
    归一化
    :param data_file:
    :param train_file:
    :param test_file:
    :return:
    """
    from pyspark.ml.feature import StandardScaler, MinMaxScaler
    sess = get_spark_sesssion()
    dataframe = sess.read.load(data_file, format=os.path.splitext(data_file)[
                                                     1][1:], header=True, inferSchema=True)

    train_frame = sess.read.load(data_file, format=os.path.splitext(train_file)[
                                                       1][1:], header=True, inferSchema=True)

    test_frame = sess.read.load(data_file, format=os.path.splitext(test_file)[
                                                      1][1:], header=True, inferSchema=True)
    columns = dataframe.columns
    for c in columns:
        if c in real_cnt_feats or c == 'action_installed':
            print(c)
            model = MinMaxScaler(outputCol='std_' + c, inputCol=c)
            model = model.fit(dataframe)
            train_frame = model.transform(train_frame)
            train_frame = train_frame.drop(c)
            train_frame = train_frame.withColumnRenamed('std_' + c, c)

            test_frame = model.transform(test_frame)
            test_frame = test_frame.drop(c)
            test_frame = test_frame.withColumnRenamed('std_' + c, c)

    save_pandas(train_frame.toPandas(), 'train_nm.csv', index=False)
    save_pandas(test_frame.toPandas(), 'test_nm.csv', index=False)


def split_cv(train_file, days_for_val=2, start=17, end=30, base_dir='./'):
    """
    交叉验证, 分割 train1 val1 train2 val2
    :param train_file:
    :param days_for_val:
    :param start:
    :param end:
    :return:
    """
    ensure_exits(base_dir)
    ss = list(range(start, end, days_for_val))
    ee = ss[1:] + [end]
    periods = zip(ss, ee)
    for i, (s, e) in enumerate(periods):
        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(
                filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] >= s) & (df['clickTime_day'] <= e), :],
            save_func=lambda df: save_pandas(
                df, base_dir + 'val_{:02}.csv'.format(i + 1), append=True)
        )

        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(
                filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] < s) | (df['clickTime_day'] > e), :],
            save_func=lambda df: save_pandas(
                df, base_dir + 'train_{:02}.csv'.format(i + 1), append=True)
        )


def split_window_cv(train_file, days_for_train=3, days_for_val=1, start=17, end=30, base_dir='./'):
    """
    交叉验证, 分割 train1 val1 train2 val2
    :param train_file:
    :param days_for_val:
    :param start:
    :param end:
    :return:
    """
    ensure_exits(base_dir)
    for i in range(end - days_for_train - days_for_val + 2):
        train_start = start + i
        train_end = train_start + days_for_train
        train_name = base_dir + 'train_{:02}.csv'.format(i + 1)

        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(
                filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] >= train_start) & (df['clickTime_day'] < train_end), :],
            save_func=lambda df: save_pandas(
                df, train_name, append=True, index=False)
        )

        val_start = train_end
        val_end = val_start + days_for_val
        val_name = base_dir + 'val_{:02}.csv'.format(i + 1)
        val_label_name = base_dir + 'val_label_{:02}.csv'.format(i + 1)

        def split_save(df):
            save_pandas(df, val_name, append=True, index=False)
            df_label = pd.DataFrame(df.loc[:, ['label']])
            save_pandas(df_label, val_label_name, append=True, index=False)  # 保存label 方便合并

        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(
                filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] >= val_start) & (df['clickTime_day'] < val_end), :],
            save_func=split_save
        )

        print('[{}, {}] [{}, {}]'.format(train_start, train_end, val_start, val_end))


def main():
    # split_cv('./result.hdf5', base_dir='cv/')
    # df_infos_summary('../train.csv')
    normalize('../total_ffm.csv', '../train_ffm.csv', '../test_ffm.csv')


if __name__ == '__main__':
    main()
