import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import sys
import time
import xgboost as xgb

import xgbfir





t0org0 = pd.read_csv("train.csv")
#h0org = pd.read_csv("test.csv")
print t0org0.columns
features=t0org0
lable=features['label']
features.drop(['label'], axis=1,inplace=True)
features.userID=features.userID.astype('int64')
features.cnt_advertiserID=features.cnt_advertiserID.astype('int64')

# features.drop(['conversionTime'], axis=1,inplace=True)

# dtrain = xgb.DMatrix(feature, label=lable, missing=-1)
# dvalid = xgb.DMatrix(xvalid, label=yvalid, missing=-1)
xgb_cmodel = xgb.XGBClassifier().fit(features, lable)

# saving to file with proper feature names

xgbfir.saveXgbFI(xgb_cmodel, feature_names=features.columns, OutputXlsxFile = 'irisFI1.xlsx')

# irisFI = [pd.read_excel("irisFI.xlsx", sheetname = "Interaction Depth %d" % i) for i in range(3)]

# one_feature_list=irisFI[0].Interaction
# for column in one_feature_list:
# 	print column, t0org0[column].unique().shape, t0org0[column].min(), t0org0[column].max()