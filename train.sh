#!/usr/bin/env bash

python train.py

python models/ffm.py fmt_file --cinfo_path column_summary.pkl --file_path ../test_ffm.csv --ffm_path test_ffm.ffm /
--for_train False

for i in {1..11}
do
echo "-------------------------"
printf -v iter '%02d' $i
python models/ffm.py fmt_file --cinfo_path column_summary.pkl --file_path cv/train_${i}.csv --ffm_path cv/ffm_train_${i}.ffm /
--for_train True
python models/ffm.py fmt_file --cinfo_path column_summary.pkl --file_path cv/val_${i}.csv --ffm_path cv/ffm_val_${i}.ffm /
--for_train True

../libffm/ffm-train -l 0.0001 -k 15 -t 300 -r 0.05 -s 4 --auto-stop -p cv/ffm_val_${i}.ffm  cv/ffm_train_${i}.ffm cv/model_${i}.ffm

../libffm/ffm-predict cv/ffm_val_${i}.ffm cv/model_${i}.ffm cv/predict_val_${i}.ffm
../libffm/ffm-predict test_ffm.ffm cv/model_${i}.ffm cv/predict_test_${i}.ffm

tee -i '1i\pred_ffm' cv/predict_val_${i}.ffm
paste -d"," val_label_${i}.csv cv/predict_val_${i}.ffm > cv/cmb-predict-val-${i}.ffm
tee -i '1d' cv/cmb-predict-val-${i}.ffm
cat cv/cmb-predict-val-${i}.ffm >> cv/pred-val-all.ffm.csv

tee -i '1i\pred_${i}' cv/predict_test_${i}.ffm
done


tee -i '1i\label, pred_ffm' cv/pred-val-all.ffm.csv
paste -d"," cv/predict-test-*.ffm >> cv/pred-test-all.ffm.csv