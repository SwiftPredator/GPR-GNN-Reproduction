CSV_NAME=cSBM_data
RPMAX=100
DATASET=cSBM_data
for model in SGC MLP JKNet APPNP GPRGNN GCN GAT ChebNet
do
    for phi in -1.0 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1
    do
        python train.py --RPMAX $RPMAX \
                --net $model \
                --train_rate 0.025 \
                --val_rate 0.025 \
                --dataset ${DATASET}_phi_${phi} \
                --csv_name ${CSV_NAME}_phi_${phi}.csv

        python train.py --RPMAX $RPMAX \
                --net $model \
                --train_rate 0.6 \
                --val_rate 0.2 \
                --dataset ${DATASET}_phi_${phi} \
                --csv_name ${CSV_NAME}_phi_${phi}.csv
    done
done