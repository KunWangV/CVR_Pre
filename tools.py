# coding: utf-8

import pandas as pd

from utils import *

map_by_chunk('../hist_features/df_hist_feature_all.csv',
             read_func=lambda x: pd.read_csv(x, iterator=True),
             map_func=lambda df: df.drop(df.columns[df.columns.str.startswith('Unnamed')], axis=1),
             save_func=lambda df: df.to_csv('../hist_features/df_hist_feature_all_clean.csv', mode='a+',
                                            float_format='.7f'))


