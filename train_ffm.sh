# !/usr/bin/bash
# python ffm.py

# 0.107/8 100
# ../libffm/ffm-train -t 200 -l 0.0001 -k 15 -r 0.1 -s 8 -p df_test.ffm df_train.ffm model 
# ../libffm/ffm-train -t 200 -l 0.0001 -k 10 -r 0.05 -s 8 -p df_test.ffm df_train.ffm model
# ../libffm/ffm-train -t 200 -l 0.0001 -k 30 -r 0.05 -s 8 -p df_test.ffm df_train.ffm model

# 0.101
# ../libffm/ffm-train -t 200 -l 0.0001 -k 15 -r 0.1 -s 8 -p df_test.week.ffm df_train.week.ffm model-week1
# ../libffm/ffm-train -t 200 -l 0.0001 -k 10 -r 0.05 -s 8 -p df_test.week.ffm df_train.week.ffm model-week2
../libffm/ffm-train -t 1000 -l 0.0001 -k 30 -r 0.05 -s 8 -p df_test.week.ffm df_train.week.ffm model-week3

# ../libffm/ffm-train -l 0.0001 -k 15 -t 30 -r 0.05 -s 8 --auto-stop -p df_test.ffm df_train.ffm model

../libffm/ffm-predict ../ffms/df_pred.ffm model output