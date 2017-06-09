# coding: utf-8

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from keras.layers import Concatenate, Conv1D, LocallyConnected1D, Dense, Dropout, Input, Embedding, concatenate
from keras.models import Sequential, Model
from keras.metrics import binary_accuracy
from keras.preprocessing.image import ImageDataGenerator

real_cvt_feats = [
]

real_cnt_feats = [

]

real_other = [
]

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
drop_feats = [

]


class PandasGenerator(object):
    """
    """
    def __next__(self):
        pass

    def __len__(self):
        pass


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
        return 'name: {}, type: {}, max value: {}, dtype: {}'.format(self.name, self.type, self.max_value, self.dtype)


def get_model(column_info_list, hidden_layers=[512, 256, 128], ):
    inputs = []
    real_inputs = []
    cate_inputs = []
    embeddings = []

    for column in column_info_list:
        if column.type == 'category':
            input = Input(shape=(1,), dtype=column.dtype, name='input_{}'.format(column.name))
            emb = Embedding(output_dim=10, input_dim=column.unique_size, input_length=1)
            embeddings.append(emb)
            cate_inputs.append(input)

        elif column.type == 'real':
            input = Input(shape=(1,), dtype=column.dtype, name='input_{}'.format(column.name))
            real_inputs.append(input)

        inputs.append(input)

    x = concatenate(embeddings + real_inputs)
    for layer_size in hidden_layers:
        x = Dense(layer_size, activation='sigmoid')(x)

    output = Dense(1, activation='softmax', name='output')

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', sample_weight_mode=[1, 1 / 0.02])
    model.fit_generator(epochs=30, validation_data=)


if __name__ == '__main__':
    main()
