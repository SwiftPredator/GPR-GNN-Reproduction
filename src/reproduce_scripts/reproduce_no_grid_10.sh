#! /bin/sh
#
# This script is to reproduce our results in Table 2.
RPMAX=10
csv_name=./nogrid_10.csv
# Below is for homophily datasets, sparse split

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.025 \
#         --val_rate 0.025 \
#         --dataset cora \
#         --lr 0.01 \
#         --alpha 0.1 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.025 \
#         --val_rate 0.025 \
#         --dataset citeseer \
#         --lr 0.01 \
#         --alpha 0.1 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.025 \
#         --val_rate 0.025 \
#         --dataset pubmed \
#         --lr 0.05 \
#         --alpha 0.2 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.025 \
#         --val_rate 0.025 \
#         --dataset computers \
#         --lr 0.05 \
#         --alpha 0.5 \
#         --weight_decay 0.0 \
#         --csv_name $csv_name     

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.025 \
#         --val_rate 0.025 \
#         --dataset photo \
#         --lr 0.01 \
#         --alpha 0.5 \
#         --weight_decay 0.0 \
#         --csv_name $csv_name

# # Below is for heterophily datasets, dense split

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.6 \
#         --val_rate 0.2 \
#         --dataset chameleon \
#         --lr 0.05 \
#         --alpha 1.0 \
#         --weight_decay 0.0 \
#         --dprate 0.7 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.6 \
#         --val_rate 0.2 \
#         --dataset film \
#         --lr 0.01 \
#         --alpha 0.9 \
#         --weight_decay 0.0 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.6 \
#         --val_rate 0.2 \
#         --dataset squirrel \
#         --lr 0.05 \
#         --alpha 0.0 \
#         --weight_decay 0.0 \
#         --dprate 0.7 \
#         --csv_name $csv_name 

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.6 \
#         --val_rate 0.2 \
#         --dataset texas \
#         --lr 0.05 \
#         --alpha 1.0 \
#         --csv_name $csv_name

# python train.py --RPMAX $RPMAX \
#         --net GPRGNN \
#         --train_rate 0.6 \
#         --val_rate 0.2 \
#         --dataset cornell \
#         --lr 0.05 \
#         --alpha 0.9 \
#         --csv_name $csv_name


for model in GCN GAT APPNP ChebNet JKNet SGC MLP
do
        for dat in cora citeseer pubmed computers photo
        do
                python train.py --RPMAX $RPMAX \
                        --net $model \
                        --train_rate 0.025 \
                        --val_rate 0.025 \
                        --dataset $dat \
                        --csv_name $csv_name
        done

        for dat in chameleon film squirrel texas cornell
        do
                python train.py --RPMAX $RPMAX \
                        --net $model \
                        --train_rate 0.6 \
                        --val_rate 0.2 \
                        --dataset $dat \
                        --csv_name $csv_name
        done

done

