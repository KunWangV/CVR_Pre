# coding: utf-8
# pylint: disable=C0103, C0111

# from pyspark.sql import SparkSession
import pandas as pd

FILE_AD = r'../ad.csv'
FILE_APP_CATEGORIES = r'../app_categories.csv'
FILE_POSITION = r'../position.csv'
FILE_SUBMISSION = r'../submission.csv'
FILE_TEST = r'../test.csv'
FILE_TRAIN = r'../train.csv'
FILE_USER_APP_ACTIONS = r'../user_app_actions.csv'
FILE_USER = r'../user.csv'
FILE_USER_INSTALLEDAPPS = r'../user_installedapps.csv'

# sess = SparkSession.builder.appName('tencent') \
#     .config('spark.executor.memory', '1024m') \
#     .config('spark.driver.memory', '1024m') \
#     .master('local[4]') \
#     .getOrCreate()


# def load_file(file_name, view_name):
#     df = sess.read.csv(file_name, inferSchema=True, header=True)
#     df.createOrReplaceTempView(view_name)
#     return sess, df


def read_as_pandas(filename):
    return pd.read_csv(filename)

#
# def load_ad():
#     return load_file(FILE_AD, 'ad')
#
#
# def load_app_categories():
#     return load_file(FILE_APP_CATEGORIES, 'app_categories')
#
#
# def load_position():
#     return load_file(FILE_POSITION, 'position')
#
#
# def load_submission():
#     return load_file(FILE_SUBMISSION, 'submission')
#
#
# def load_test():
#     return load_file(FILE_TEST, 'test')
#
#
# def load_train():
#     return load_file(FILE_TRAIN, 'train')
#
#
# def load_user_app_actions():
#     return load_file(FILE_USER_APP_ACTIONS, 'user_app_actions')
#
#
# def load_user():
#     return load_file(FILE_USER, 'user')
#
#
# def load_user_installedapps():
#     return load_file(FILE_USER_INSTALLEDAPPS, 'user_installedapps')


# if __name__ == '__main__':
#     # ss, ad = load_ad()
#     print ad.head(5)
#     mm = ss.sql("select adID from ad")
#     print mm.toPandas()
#     print read_as_pandas(FILE_AD).head(5)
    