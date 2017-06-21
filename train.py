# coding:utf-8
from data import *
from utils import *
from config import *

"""
总的训练模型 走流程
"""

train_file = '../train_dc.csv'
test_file = '../test.csv'

split_window_cv(train_file, days_for_val=1, days_for_train=3, base_dir='cv/')
