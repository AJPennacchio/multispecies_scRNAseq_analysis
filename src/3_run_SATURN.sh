#!/bin/bash

# $1 = PATH to train-saturn.py
# $2 = PATH to csv file
# $3 = h5 column name to guide SATURN's integration (cell type labels or leiden, depending on usage)
# $4 = h5 column name to reference labels, optional
# note: can adjust hv_genes and num_macrogenes, these are SATURN defaults

python3 $1 --in_data=$2 --in_label_col=$3 --ref_label_col=$4 --hv_genes=8000 --num_macrogenes=2000
