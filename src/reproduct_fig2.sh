#! /bin/sh
#
# This script is to reproduce our results in Table 2.
RPMAX=100
csv_name=./fig2_1epsilon.csv
# Below is for homophily datasets, sparse split
dataset=cSBM_data_1


for model in GCN GAT ChebNet JKNet SGC MLP APPNP GPRGNN
do
    python train.py --RPMAX $RPMAX \
            --net $model \
            --train_rate 0.025 \
            --val_rate 0.025 \
            --dataset $dataset \
            --csv_name $csv_name 

    python train.py --RPMAX $RPMAX \
            --net $model \
            --train_rate 0.6 \
            --val_rate 0.2 \
            --dataset $dataset \
            --csv_name $csv_name 
done
