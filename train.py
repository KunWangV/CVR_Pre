# coding:utf-8
from data import *
from utils import *
from config import *

"""
总的训练模型 走流程
"""

train_file = '../train_ffm.csv'
test_file = '../test_ffm.csv'
# train_file = '../train.csv'
# test_file = '../test.csv'
merged_file = '../total_ffm.csv'

# merge_txt([train_file, test_file], merged_file, skip_header=True)  没用 忘记了train和test结构不一直

# df_train = read_as_pandas(train_file, chunk_size=None)
# print df_train.shape
# df_test = read_as_pandas(test_file, chunk_size=None)
# print df_test.shape
# df_merged = pd.concat([df_train, df_test])
# save_pandas(df_merged, merged_file, index=False)

# generate summary
df_infos_summary(merged_file, save_name='column_summary.pkl')

split_window_cv(train_file, days_for_val=1, days_for_train=3, base_dir='cv/')
