#!/bin/sh
cd `dirname $0`
cd ..

# common settngs
INPUT="Engine"
OUTPUT="_out/Engine"

window_size=20
feature_cols="sensor2/sensor3/sensor4/sensor7/sensor11/sensor12/sensor15"
label_col="RUL"
import_type="clean_data_rul_k_folds_cmapss_condition1"
split_ind=0


alpha=1
beta_w=0.1
k=5
niter=100
tag="demo"
python3 -u _src/timecast/main.py  --input_tag $INPUT \
                                 --out_dir $OUTPUT"/timecast/w_"$window_size"/alpha_"$alpha"/beta_w_"$beta_w"/k_"$k"/maxiter_"$niter"/folds_"$split_ind"/tag_"$tag \
                                 --feature_cols $feature_cols \
                                 --label_col $label_col \
                                 --window_size $window_size \
                                 --import_type $import_type \
                                 --split_ind $split_ind \
                                 --use_sequence_inputs \
                                 \
                                 --alpha $alpha \
                                 --beta_w $beta_w \
                                 --init_k $k \
                                 --niter $niter