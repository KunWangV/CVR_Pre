# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
from keras.layers import Dense, Dropout, Input, Embedding, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import loss
from config import *
from feature.data import *
from utils import *

callback_logloss = []

from argparse import ArgumentParser


class PandasGenerator(object):
    """
    一次读取
    """

    def __init__(self,
                 df_x,
                 df_y,
                 batch_size,
                 infos,
                 shuffle=True,
                 with_weight=False,
                 for_train=False):
        self.column_list = []
        for info in infos:
            self.column_list.append(info.name)

        self.df_x = pd.read_csv(df_x).loc[:, self.column_list]

        self.df_y = None
        if df_y is not None:
            self.df_y = pd.read_csv(df_y)

        if for_train:  # select on week
            print(
                'for train select ',
                self.df_x.shape, )
            self.df_x = self.df_x.loc[self.df_x['clickTime_day'] > 16, :]
            if self.df_y is not None:
                self.df_y = self.df_y.loc[self.df_x.index, :]
            print('after select ', self.df_x.shape)

        self.df_x['age'] = self.df_x['age'] // 5  #
        self.batch_size = batch_size

        self.length = self.df_x.shape[0]
        self.idx = 0

        if batch_size == 'all':
            self.batch_size = self.length

        if batch_size == 'auto':
            self.batch_size = self.max_batch_size()

        self.seq = range(self.length)
        self.shuffle = shuffle
        self.with_weight = with_weight
        if shuffle:
            np.random.shuffle(self.seq)

        print(self.df_x.shape)
        if self.df_y is not None:
            print(self.df_y.shape)

        print(self.df_x.clickTime_day.unique())

    def next(self):

        if self.idx + self.batch_size <= self.length:
            end = self.idx + self.batch_size
            x = self.df_x.iloc[self.seq[self.idx:end], :].values.copy()
            if self.df_y is not None:
                y = self.df_y.iloc[self.seq[self.idx:end], :].values.copy()

            self.idx += self.batch_size

        else:
            x = self.df_x.iloc[self.seq[self.idx:self.length], :].values.copy()
            if self.df_y is not None:
                y = self.df_y.iloc[self.seq[self.idx:
                self.length], :].values.copy()

            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.seq)

        # print(np.max(y))
        inputs = np.split(x, x.shape[1], axis=1)
        if self.df_y is not None:
            outputs = to_categorical(y, 2)
            outputs = outputs[:, np.newaxis, :]

            if self.with_weight:
                weights = np.squeeze(y)
                weights[weights == 1] = 1 / 0.026  # sample weight
                weights[weights == 0] = 1

                return inputs, outputs, weights
            else:
                return inputs, outputs

        return inputs

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def n_per_epoch(self):
        return np.ceil(self.length / float(self.batch_size))

    def label(self):
        if self.df_y is not None:
            return self.df_y.values

        return None

    def max_batch_size(self):
        i = 2
        for i in range(1000, 100):
            if self.length % i == 0:
                break

        return self.length // i


class PandasChunkGenerator():
    """
    """

    def __init__(self,
                 infos,
                 filename,
                 total_records,
                 batch_size=100000,
                 for_train=True,
                 label_column='label',
                 with_weight=False):
        self.columns = get_columns_from_column_infos(infos)
        self.reader = PandasChunkReader(filename, loop=True, chunk_size=batch_size)
        self.for_train = for_train
        self.label_column = label_column
        self.length = total_records
        self.idx = 0
        self.df_y = None
        self.with_weight = with_weight
        self.batch_size = batch_size

    def next(self):
        self.idx += 1
        if self.idx > self.n_per_epoch():
            self.df_y = None

        df = self.reader.next()
        x = df.loc[:, self.columns]
        x = x.values

        if self.for_train:
            y = df[self.label_column]
            if self.df_y is None:
                self.df_y = y

            else:
                self.df_y = pd.concat([self.df_y, y], axis=0)

            y = y.values

        inputs = np.split(x, x.shape[1], axis=1)
        if self.for_train:
            outputs = to_categorical(y, 2)
            outputs = outputs[:, np.newaxis, :]

            if self.with_weight:
                weights = np.squeeze(y)
                weights[weights == 1] = 1 / 0.026  # sample weight
                weights[weights == 0] = 1

                return inputs, outputs, weights
            else:
                return inputs, outputs

        return inputs

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def n_per_epoch(self):
        return np.ceil(self.length / float(self.batch_size))

    def label(self):
        if self.df_y is not None:
            return self.df_y.values

        return None

    def max_batch_size(self):
        i = 2
        for i in range(1000, 100):
            if self.length % i == 0:
                break

        return self.length // i


class EvalCallback(Callback):
    """
    计算logloss
    """

    def __init__(self, model, generator):
        super(EvalCallback, self).__init__()
        self.model = model
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict_generator(self.generator,
                                            self.generator.n_per_epoch())
        print(pred.shape)
        ppred = np.squeeze(pred[:, :, 1])
        ll = loss.logloss(np.squeeze(self.generator.label()), ppred)
        callback_logloss.append(ll)


def make_model(column_info_list,
               hidden_layers=[512, 256, 128]):
    inputs = []
    real_inputs = []
    cate_inputs = []
    embeddings = []

    for column in column_info_list:
        if column.type == 'category':
            input = Input(
                shape=(1,),
                dtype=column.dtype,
                name='input_{}'.format(column.name))
            emb = Embedding(
                output_dim=10,
                input_dim=column.unique_size,
                input_length=1,
                embeddings_regularizer=regularizers.l1_l2(0.0001), )(input)
            embeddings.append(emb)
            cate_inputs.append(input)

        elif column.type == 'real':
            input = Input(
                shape=(1,),
                dtype=column.dtype,
                name='input_{}'.format(column.name))
            real_inputs.append(input)

        inputs.append(input)

    x = concatenate(embeddings + real_inputs)
    for layer_size in hidden_layers:
        x = Dense(
            layer_size,
            activation='sigmoid',
            activity_regularizer=regularizers.l1(0.0001),
            kernel_regularizer=regularizers.l2(0.0001), )(x)
        x = Dropout(0.8)(x)

    output = Dense(
        2,
        activation='softmax',
        name='output',
        activity_regularizer=regularizers.l1(0.0001),
        kernel_regularizer=regularizers.l2(0.0001), )(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    keras.utils.plot_model(model, to_file='model.png')
    return model


def train(model, batch_size, column_info_list, gen_train, gen_val):
    optimizer = Adam(lr=0.001)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            filepath='dnn-model/weights.{epoch:02d}.hdf5',
            monitor='val_loss',
            period=1))
    callbacks.append(TensorBoard(log_dir='./.logs', histogram_freq=1))
    callbacks.append(
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001))

    callbacks.append(EvalCallback(model, gen_val))

    model.fit_generator(
        epochs=30,
        generator=gen_train,
        validation_data=gen_val,
        validation_steps=gen_val.n_per_epoch(),
        steps_per_epoch=gen_train.n_per_epoch(),
        # steps_per_epoch=10,
        callbacks=callbacks, )


def predict(model, gen_pre, predict_name):
    """
    找最好的预测
    :param model:
    :param gen_pre:
    :return:
    """
    lls = np.asarray(callback_logloss)
    idx = lls.argmax()
    model.load_weights('dnn-model/weights.{epoch:02d}.hdf5'.format(idx + 1))
    results = model.predict_generator(gen_pre, steps=gen_pre.n_per_epoch())
    df_result = pd.DataFrame(results.reshape(-1, 1))
    df_result.columns = ['pred']
    df_result['instanceID'] = range(1, df_result.shape[0] + 1)
    df_result.to_csv(predict_name, index=False)


#
# def main():
#     infos = load_pickle(filename=COLUMN_LIST_FILENAME)
#     print('train....')
#
#     batch_size = 10000
#     gen_train = PandasChunkGenerator(
#         infos,
#         'train.hdf5',
#         30000000,
#     )
#
#     gen_val = PandasChunkGenerator(
#         infos,
#         'val.hdf5',
#         30000000,
#     )
#
#     model = make_model(infos, hidden_layers=[512, 512, 512])
#     train(model, batch_size, infos, gen_train, gen_val)
#
#     gen_test = PandasChunkGenerator(
#         infos,
#         'test.hdf5',
#         30000000,
#     )
#     predict(model, gen_test)


# def main_old(args):
# infos = load_pickle(filename=COLUMN_LIST_FILENAME)
# print('train....')
#
# batch_size = 10000
#
# gen_train = PandasGenerator(
#     'df_trainx.csv',
#     'df_trainy.csv',
#     batch_size,
#     infos,
#     shuffle=True,
#     with_weight=False,
#     for_train=True)
#
# gen_test = PandasGenerator(
#     'df_testx.csv',
#     'df_testy.csv',
#     batch_size,
#     infos,
#     shuffle=False,
#     with_weight=False,
#     for_train=False)
#
# gen_train = PandasChunkGenerator(
#     infos,
#     'train.hdf5',
# )
# model = make_model(infos, hidden_layers=[512, 512, 512])
# train(model, batch_size, infos, gen_train, gen_test)
#
# gen_pre = PandasGenerator(
#     'df_basic_test.csv',
#     None,
#     batch_size,
#     infos,
#     shuffle=False,
#     with_weight=False,
#     for_train=False)
#
# predict(model, gen_pre)

def main(args):
    infos = load_pickle(args.cinfo_path)

    batch_size = args.batch_size
    gen_train = PandasChunkGenerator(
        infos,
        args.train_path,
        args.train_file_lines,
    )

    gen_val = PandasChunkGenerator(
        infos,
        args.val_path,
        args.val_file_lines,
    )

    print('train....')
    model = make_model(infos, hidden_layers=args.hidden_layers)
    train(model, batch_size, infos, gen_train, gen_val)

    gen_test = PandasChunkGenerator(
        infos,
        args.test_path,
        args.test_file_lines,
    )
    print('predicting ...')
    predict(model, gen_test, args.predict_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cinfo_path', default=COLUMN_LIST_FILENAME)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--train_path', default='train.hdf5')
    parser.add_argument('--train_file_lines', require=True, type=int)
    parser.add_argument('--val_path', default='val.hdf5')
    parser.add_argument('--val_file_lines', require=True, type=int)
    parser.add_argument('--hidden_layers', type=list, default=[512, 512, 512])
    parser.add_argument('--test_path', default='test.hdf5')
    parser.add_argument('--test_file_lines', require=True, type=int)
    parser.add_argument('--predict_name', default='submission.dnn.csv')

    args = parser.parse_args()
    main(args)
