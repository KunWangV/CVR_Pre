# coding: utf-8
from __future__ import print_function
from utils import *


def spark_gbdt(train_file, test_file, features_columns='userID'):
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
    sess = get_spark_sesssion()

    string_indexer = StringIndexer(inputCol="label", outputCol="idx_label")
    v_c = VectorAssembler(inputCols=['userID'], outputCol='v_userID')

    gbdt = GBTClassifier(maxDepth=5, labelCol="idx_label",
                         predictionCol="prediction",
                         featuresCol='v_userID', seed=42,
                         maxMemoryInMB=1024 * 4,
                         maxIter=4)

    train = sess.read.load(train_file, format='csv',
                           header=True, inferSchema=True, )

    train_data = string_indexer.fit(train).transform(train)
    train_data = v_c.transform(train_data)
    train_data.printSchema()

    model = gbdt.fit(train_data)
    model.save('gbtc.model')
    print(model.featureImportances)

    test = sess.read.load(test_file, format='csv',
                          header=True, inferSchema=True)

    test_data = string_indexer.fit(test).transform(test)
    test_data = v_c.transform(test_data)
    predict = model.transform(test_data, labelCol="idx_label",
                              predictionCol="prediction",
                              featuresCol='v_userID',)

    predict.show()

if __name__ == '__main__':
    spark_gbdt('../train.csv', '../test.csv')
