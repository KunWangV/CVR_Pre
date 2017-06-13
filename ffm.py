# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pickle
from feature.data import *
from feature.common import *

real_cvt_feats = []

real_cnt_feats = []

real_other = []

cate_low_dim = [
    'age',
    'appCategory',
    'appPlatform',
    'clickTime_day',
    'clickTime_hour',
    'clickTime_minute',
    'clickTime_week',
    'clickTime_seconds',
    'connectionType',
    'education',
    'gender',
    'haveBaby',
    'hometown_c',
    'hometown_p',
    'marriageStatus',
    'positionType',
    'telecomsOperator',
    'residence_c',
    'residence_p',
    'appID',
    'sitesetID',
]

cate_high_dim = [
    'adID',
    'advertiserID',
    'camgaignID',
    'creativeID',
    'positionID',
    'userID',
]

cate_feats = cate_high_dim + cate_low_dim
real_feats = real_cnt_feats + real_cvt_feats + real_other
drop_feats = [
    'userID',
]

all_feats = cate_feats + real_feats

infos = load_pickle('column_list.pkl')


def gen_file(df_x_path, df_y_path, out_filename, test=False, for_train=False):
    df_x_reader = pd.read_csv(df_x_path, iterator=True)
    if not test:
        df_y_reader = pd.read_csv(df_y_path, iterator=True)

    loop = True
    chunk_size = 100000
    idx = 0
    while loop:
        idx += 1
        print('>>>>>>', idx)
        try:
            df_x = df_x_reader.get_chunk(chunk_size)
            if test:
                del df_x['instanceID']
                del df_x['label']

            if not test:
                df_y = df_y_reader.get_chunk(chunk_size)
                print(df_y.shape)

            print(df_x.shape, df_x['clickTime_day'].unique())
            if for_train:
                df_x = df_x.loc[df_x['clickTime_day'] >=23, :]
                if not test:
                    df_y = df_y.loc[df_x.index, :]
                    print(df_y.shape)

            print(df_x.shape, df_x['clickTime_day'].unique())

            if df_x.shape[0] > 0:
                df_x['age'] = df_x['age'].astype(int) // 5
                print(df_x.shape)
                feat_idx = 0
                for field_idx, info in enumerate(infos):
                    if info.name in drop_feats:
                        continue

                    print(
                        info.name, )
                    if info.type == 'category':
                        df_x[info.name] = df_x[info.name] + feat_idx
                        df_x[info.name] = "{}:".format(
                            field_idx) + df_x[info.name].astype(str) + ':1'
                        feat_idx += info.unique_size + 1
                    else:
                        df_x[info.name] = "{}:{}:".format(
                            field_idx, feat_idx) + df_x[info.name].astype(str)
                        feat_idx += 1

                if test:
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
        except StopIteration:
            loop = False
            print("Iteration is stopped.")


if __name__ == '__main__':
    gen_file(
        'df_trainx.csv',
        'df_trainy.csv',
        'df_train.week.ffm',
        False,
        for_train=True)
    gen_file(
        'df_testx.csv',
        'df_testy.csv',
        'df_test.week.ffm',
        False,
        for_train=True)
    # gen_file(
    #     'df_basic_test.csv',
    #     None,
    #     'df_pred.ffm',
    #     test=True,
    #     for_train=False, )
