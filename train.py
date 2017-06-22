# coding:utf-8
from data import *
from utils import *
from config import *

"""
总的训练模型 走流程
"""

train_file = '../train_ffm.csv'
test_file = '../test_ffm.csv'
merged_file = '../total_ffm.csv'

merge_txt([train_file, test_file], merged_file, skip_header=True)

# generate summary
df_infos_summary(merged_file, save_name='column_summary.pkl')

split_window_cv(train_file, days_for_val=1, days_for_train=3, base_dir='cv/')
