# coding: utf-8

from feature.common import *
import pandas as pd

map_by_chunk('../hist_features/df_hist_feature_all.csv',
             read_func=lambda x: pd.read_csv(x),
             map_func=lambda df: df.drop(df.columns[df.columns.str.startswith('Unnamed')], axis=1),
             save_func=lambda df: df.to_csv('../hist_features/df_hist_feature_all_clean.csv', mode='a+',
                                            float_format='.7f'))


