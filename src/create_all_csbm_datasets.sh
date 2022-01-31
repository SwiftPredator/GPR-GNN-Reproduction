#! /bin/sh
#
# create_cSBM_dataset.sh

for phi in -1.0 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1
do
    python cSBM_dataset.py --phi $phi \
        --name cSBM_data_phi_${phi} \
        --root ../data/ \
        --num_nodes 5000 \
        --num_features 2000 \
        --avg_degree 5 \
        --epsilon 3.25 

done