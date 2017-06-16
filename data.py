# coding: utf-8
from utils import *


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
    ss = range(start, end, 2)
    ee = ss[1:] + [end]
    periods = zip(ss, ee)
    for i, (s, e) in enumerate(periods):
        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] >= s) & (df['clickTime_day'] <= e), :],
            save_func=lambda df: save_pandas(
                df, base_dir + 'val{:02}.hdf5'.format(i + 1), append=True)
        )

        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(filename, by_chunk=True, chunk_size=100000),
            map_func=lambda df: df.loc[
                                (df['clickTime_day'] < s) | (df['clickTime_day'] > e), :],
            save_func=lambda df: save_pandas(
                df, base_dir + 'train{:02}.hdf5'.format(i + 1), append=True)
        )


def main():
    split_cv('./result.hdf5', base_dir='cv/')


if __name__ == '__main__':
    main()
