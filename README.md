# MODIG: Integrating Multi-Omics and Multi-Dimensional Gene Network for Cancer Driver Gene Identification based on Graph Attention Network Model


## Introduction
A graph attention network-based model by combining multi-omics data, such as mutations, copy number variants, DNA methylation, and gene expression, together with multi-dimensional gene networks

## Methods
### modig_graph.py 
- generate multi-dimensional gene networks as input.
### modig.py
- the implementation of MODIG.
### main.py
- the main script of MODIG.
### utils.py
- functions used in MODIG.

## Data fold contains the following data:
- labels for training
- omics data
- ppi network
- gene association profiles

## Environments
- Python 3.8
- Pytorch 1.8.1+cu102
- Pytorch Geometric 2.0.0
- networkx 2.6.3
- numpy 1.21.2
- scipy 1.7.1
- sklearn 0.24.2
- pandas 1.3.3
