# coding: utf-8

from __future__ import print_function
from __future__ import division

import pandas as pd
import pickle
import numpy as np
import os

from feature.data import *

from keras.layers import Concatenate, Conv1D, LocallyConnected1D, Dense, Dropout, Input, Embedding, concatenate
from keras.models import Sequential, Model
from keras.metrics import binary_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

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

    def __init__(self, df_x, df_y, batch_size, infos):
        column_list = []
        for info in infos:
            column_list.append(info.name)

        self.df_x = pd.read_csv(df_x).loc[:, column_list]
        self.df_y = pd.read_csv(df_y)
        self.batch_size = batch_size
        self.length = self.df_x.shape[0]
        self.idx = 0

        print(self.df_x.shape)
        print(self.df_y.shape)

    def next(self):
        if self.idx + self.batch_size <= self.length:
            x = self.df_x.iloc[self.idx:self.idx + self.batch_size].values
            y = self.df_y.iloc[self.idx:self.idx + self.batch_size].values
            self.idx = (self.idx + self.batch_size) % self.length

        else:
            x = self.df_x.iloc[self.idx:self.length]
            y = self.df_y.iloc[self.idx:self.length]
            left = self.idx + self.batch_size - self.length
            _x = self.df_x.iloc[:left]
            _y = self.df_y.iloc[:left]

            x = pd.concat([x, _x], axis=0).values
            y = pd.concat([y, _y], axis=0).values
            self.idx = left

        # print(x.shape)
        # print(y.shape)
        inputs = np.split(x, x.shape[1], axis=1)
        outputs = to_categorical(y, 2)
        outputs = outputs[:, np.newaxis, :]
        return inputs, outputs

    def __iter__(self):
        return self

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
                output_dim=10, input_dim=column.unique_size + 1, input_length=1)(input)
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

    output = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=output)

    model.summary()

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    gen_train = PandasGenerator(
        'df_trainx.csv', 'df_trainy.csv', batch_size, column_info_list)
    gen_test = PandasGenerator(
        'df_testx.csv', 'df_testy.csv', 1, column_info_list)

    callbacks = []
    callbacks.append(ModelCheckpoint(
        filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', period=10))
    model.fit_generator(
        generator=gen_train,
        validation_data=gen_test,
        validation_steps=len(gen_test),
        steps_per_epoch=10473,
        # callbacks=callbacks,
    )


def gen_column_list(df, save=True, save_name='column_list.pkl'):
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


def main():
    if os.path.exists('column_list.pkl'):
        infos = pickle.load(open('column_list.pkl', 'rb'))

    else:
        print("generate column list")
        df = pd.read_csv('df_basic_train.csv')
        del df['label']
        infos = gen_column_list(df)

    print('train....')
    get_model(infos, hidden_layers=[512, 256, 128], batch_size=3000)


if __name__ == '__main__':
    main()
