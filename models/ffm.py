# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('.')

from argparse import ArgumentParser

from config import *
from utils import *
from data import *


# def gen_file(df_x_path, df_y_path, out_filename, test=False, for_train=False):
#     df_x_reader = pd.read_csv(df_x_path, iterator=True)
#     if not test:
#         df_y_reader = pd.read_csv(df_y_path, iterator=True)
#
#     loop = True
#     chunk_size = 100000
#     idx = 0
#     while loop:
#         idx += 1
#         print('>>>>>>', idx)
#         try:
#             df_x = df_x_reader.get_chunk(chunk_size)
#             if test:
#                 del df_x['instanceID']
#                 del df_x['label']
#
#             if not test:
#                 df_y = df_y_reader.get_chunk(chunk_size)
#                 print(df_y.shape)
#
#             print(df_x.shape, df_x['clickTime_day'].unique())
#             if for_train:
#                 df_x = df_x.loc[df_x['clickTime_day'] >= 23, :]
#                 if not test:
#                     df_y = df_y.loc[df_x.index, :]
#                     print(df_y.shape)
#
#             print(df_x.shape, df_x['clickTime_day'].unique())
#
#             if df_x.shape[0] > 0:
#                 df_x['age'] = df_x['age'].astype(int) // 5
#                 print(df_x.shape)
#                 feat_idx = 0
#                 for field_idx, info in enumerate(infos):
#                     if info.name in drop_feats:
#                         continue
#
#                     print(
#                         info.name, )
#                     if info.type == 'category':
#                         df_x[info.name] = df_x[info.name] + feat_idx
#                         df_x[info.name] = "{}:".format(
#                             field_idx) + df_x[info.name].astype(str) + ':1'
#                         feat_idx += info.unique_size + 1
#                     else:
#                         df_x[info.name] = "{}:{}:".format(
#                             field_idx, feat_idx) + df_x[info.name].astype(str)
#                         feat_idx += 1
#
#                 if test:
#                     df_y = pd.DataFrame(
#                         np.ones((df_x.shape[0], 1), dtype='int') * -1,
#                         columns=['label'])
#
#                 df_x.drop(drop_feats, axis=1, inplace=True)
#                 df_result = pd.concat([df_y, df_x], axis=1)
#                 print('shape to write', df_result.shape)
#                 df_result.to_csv(
#                     out_filename,
#                     mode='a+',
#                     header=False,
#                     index=False,
#                     sep=' ')
#         except StopIteration:
#             loop = False
#             print("Iteration is stopped.")


def gen_file(df_path, out_filename, infos, for_train=False, chunk_size=100000):
    """

    :param df_path:  path of dataframe with label
    :param out_filename:
    :param infos:
    :param for_train:
    :param chunk_size:
    :return:
    """

    df_reader = pd.read_csv(df_path, iterator=True, chunksize=chunk_size)
    loop = True
    idx = 0
    df_y = None
    columns = get_columns_from_column_infos(infos)
    for df_x in df_reader:
        idx += 1
        print('>>>>>>', idx)
        if for_train:
            df_y = df_x['label']

        df_x = df_x.loc[:, columns]

        if df_x.shape[0] > 0:
            df_x['age'] = df_x['age'].astype(int) // 5
            print(df_x.shape)
            feat_idx = 0
            for field_idx, info in enumerate(infos):
                if info.name in drop_feats:
                    continue

                print(info.name, )
                if info.type == 'category':
                    df_x[info.name] = df_x[info.name] + feat_idx
                    df_x[info.name] = "{}:".format(
                        field_idx) + df_x[info.name].astype(str) + ':1'
                    feat_idx += info.unique_size + 1
                else:
                    df_x[info.name] = "{}:{}:".format(
                        field_idx, feat_idx) + df_x[info.name].astype(str)
                    feat_idx += 1

            if not for_train:
                df_y = pd.DataFrame(
                    np.ones((df_x.shape[0], 1), dtype='int') * -1,
                    columns=['label'])

            df_x.drop(drop_feats, axis=1, inplace=True)
            df_result = pd.concat([df_y, df_x], axis=1)
            print('shape to write', df_result.shape)
            df_result.to_csv(
                out_filename,
                mode='a+',
                header=False,
                index=False,
                sep=' ')


def main(args):
    if args.ops == 'fmt_file':
        infos = load_pickle(args.cinfo_path)
        gen_file(args.file_path, args.ffm_path, infos, for_train=args.for_train, chunk_size=args.chunk_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    sp = parser.add_subparsers(dest='ops')
    p_file = sp.add_parser('fmt_file')
    p_file.add_argument('--cinfo_path', default=COLUMN_LIST_FILENAME)
    p_file.add_argument('--file_path')
    p_file.add_argument('--ffm_path')
    p_file.add_argument('--for_train', type=bool)
    p_file.add_argument('--chunk_size', type=int, default=100000)

    args = parser.parse_args()
    main(args)
