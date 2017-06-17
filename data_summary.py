# coding: utf-8

import pandas as pd
from utils import *
import os
import numpy as np
from argparse import ArgumentParser


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

    print shape, type(shape)

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
