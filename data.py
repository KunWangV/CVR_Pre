# coding: utf-8
from utils import *


def split_cv(train_file, days_for_val=2, start=17, end=30):
    """
    交叉验证, 分割 train1 val1 train2 val2
    :param train_file:
    :param days_for_val:
    :param start:
    :param end:
    :return:
    """
    ss = range(start, end, 2)
    ee = range[1:] + [end]
    periods = zip(ss, ee)
    for i, (s, e) in enumerate(periods):
        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(filename, by_chunk=True),
            map_func=lambda df: df.loc[(df['clickTime_day'] >= s) & (df['clickTime_day'] <= e), :],
            save_func=lambda df: save_pandas(df, 'val{02d}.hdf5'.format(i + 1), append=True)
        )

        map_by_chunk(
            train_file,
            read_func=lambda filename: read_as_pandas(filename, by_chunk=True),
            map_func=lambda df: df.loc[(df['clickTime_day'] <s) | (df['clickTime_day'] > e), :],
            save_func=lambda df: save_pandas(df, 'train{02d}.hdf5'.format(i + 1), append=True)
        )


def main():
    pass


if __name__ == '__main__':
    pass
