# coding: utf-8
# pylint: disable=C0103, C0111

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
from pyspark.ml.feature import OneHotEncoder as OHE
import pyspark.sql.functions as F

import data


def basic_features(train=True):
    sess, df_train = data.load_train()
    df_train.show(n=5)

    _, df_ad = data.load_ad()
    df_ad.show(5)

    _, df_user = data.load_user()
    df_user.show(5)

    _, df_position = data.load_position()
    df_position.show(5)

    df = df_train.join(
        df_ad, on='creativeID', how="left_outer").select("*").toDF('all')
    df.show(5)


basic_features()
