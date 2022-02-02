# GPR-GNN-Reproduction
This is a reproduction of the paper [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP).
## Installation

Our environment can be installed easily with conda. Just call the following command in the working directory:

```bash
conda env create -f environment.yml
```

## Reproduction
To reproduce our results you can run the provided bash files in the `src` directory:

```bash
./create_all_csbm_datasets.sh
```

```bash
./reproduce_no_grid_100_exp.sh
```

```bash
./reproduce_fig2.sh
```

## Paper
Details about our approach for reproducing the original paper results can be found in [our own paper](reproduction_paper.pdf). 
