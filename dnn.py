# coding: utf-8

from __future__ import print_function
from __future__ import division

import pandas as pd
import pickle
import numpy as np

from feature.data import *

from keras.layers import Concatenate, Conv1D, LocallyConnected1D, Dense, Dropout, Input, Embedding, concatenate
from keras.models import Sequential, Model
from keras.metrics import binary_accuracy
from keras.preprocessing.image import ImageDataGenerator

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
    'inst_is_installed',
    'is_day_rpt_first_click',
    'is_day_rpt_last_click',
    'is_rpt_first_click',
    'is_rpt_last_click',
    'tt_is_installed',
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
drop_feats = []


class ColumnInfo(object):
    """
    每列信息类
    """

    def __init__(self, name, type, unique_size=None, dtype='int64'):
        """
        :param name: 列名
        :param type:  category or real
        :param max_value: 最大值
        :param dtype: 数据类型
        """
        self.name = name
        self.type = type
        self.unique_size = unique_size
        self.dtype = dtype

    def __str__(self):
        return 'name: {}, type: {}, max value: {}, dtype: {}'.format(
            self.name, self.type, self.max_value, self.dtype)


class PandasGenerator(object):
    """
    """

    def __init__(self, df_x, df_y, batch_size):
        self.df_x = pd.read_csv(df_x)
        self.df_y = pd.read_csv(df_y)
        self.batch_size = batch_size
        self.length = df_x.shape[1]
        self.idx = 0

    def __next__(self):
        if self.idx + self.batch_size <= self.length:
            x = self.df_x.iloc[self.idx:self.idx + self.batch_size]
            y = self.df_y.iloc[self.idx:self.idx + self.batch_size]
            self.idx = self.idx + self.batch_size

        else:
            x = self.df_x.iloc[self.idx:self.length]
            y = self.df_y.iloc[self.idx:self.length]
            left = self.idx + self.batch_size - self.length
            _x = self.df_x.iloc[:left]
            _y = self.df_y.iloc[:left]

            x = pd.concat([x, _x], axis=0).values
            y = pd.concat([y, _y], axis=0).values
            self.idx = left

        inputs = np.split(x, x.shape[1], axis=1)
        outputs = y
        return inputs, outputs

    def __len__(self):
        return self.length


def get_model(column_info_list, hidden_layers=[512, 256, 128],
              batch_size=3000):
    inputs = []
    real_inputs = []
    cate_inputs = []
    embeddings = []

    for column in column_info_list:
        if column.type == 'category':
            input = Input(
                shape=(1, ),
                dtype=column.dtype,
                name='input_{}'.format(column.name))
            emb = Embedding(
                output_dim=10, input_dim=column.unique_size, input_length=1)
            embeddings.append(emb)
            cate_inputs.append(input)

        elif column.type == 'real':
            input = Input(
                shape=(1, ),
                dtype=column.dtype,
                name='input_{}'.format(column.name))
            real_inputs.append(input)

        inputs.append(input)

    x = concatenate(embeddings + real_inputs)
    for layer_size in hidden_layers:
        x = Dense(layer_size, activation='sigmoid')(x)

    output = Dense(1, activation='softmax', name='output')

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        sample_weight_mode=[1, 1 / 0.02])

    gen_train = PandasGenerator(
        'df_trainx.csv', 'df_trainy.csv', batch_size=batch_size)
    gen_test = PandasGenerator(
        'df_testx.csv', 'df_testy.csv', batch_size=1)
    model.fit_generator(
        generator=gen_train,
        validation_data=gen_test,
        validation_steps=len(gen_test),
        steps_per_epoch=np.floor(len(gen_train) / batch_size), )


def gen_column_list(df, save=True, save_name='column_list.pkl'):
    columns = df.columns.values
    infos = []
    for c in columns:
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

    print(infos)

    if save:
        pickle.dump(infos, open(save_name, 'wb'))

    return infos

def main():
    df = pd.read_csv('df_basic_train.csv')
    del df['label']
    infos = get_model(df)
    get_model(infos,hidden_layers=[512, 256, 128], batch_size=3000)

if __name__ == '__main__':
    main()
