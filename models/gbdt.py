# coding: utf-8
from __future__ import print_function
from utils import *


def spark_gbdt(train_file, test_file, features_columns='userID'):
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
    from pyspark.ml.pipeline import Pipeline
    sess = get_spark_sesssion()

    string_indexer = StringIndexer(inputCol="label", outputCol="idx_label")
    v_c = VectorAssembler(inputCols=['userID'], outputCol='v_userID')
    trans = Pipeline(stages=[string_indexer, v_c])

    gbdt = GBTClassifier(
        maxDepth=5,
        labelCol="idx_label",
        predictionCol="pred",
        featuresCol='v_userID',
        seed=42,
        maxMemoryInMB=1024 * 10,
        maxIter=4)

    train = sess.read.load(
        train_file,
        format='csv',
        header=True,
        inferSchema=True, )

    train_data = trans.fit(train).transform(train)
    model = gbdt.fit(train_data)
    model.write().overwrite().save('gbtc.model')
    # model = GBTClassifier.load('gbtc.model')
    print(model.featureImportances)

    test = sess.read.load(
        test_file, format='csv', header=True, inferSchema=True)

    test_data = trans.fit(test).transform(test)
    predict = model.transform(test_data)
    predict.show()

    save_pandas(
        predict.select('instanceID', 'pred').toPandas(), 'submission.gbdt.csv', index=False)


if __name__ == '__main__':
    spark_gbdt('../train.csv', '../test.csv')
