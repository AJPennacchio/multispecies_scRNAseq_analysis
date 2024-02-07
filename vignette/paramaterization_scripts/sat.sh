#!/bin/bash

python3 /home/apennacchio/SATURN/train-saturn.py --in_data=$3 --in_label_col=leiden --ref_label_col=lab --hv_genes=$2 --score_adata --ct_map_path=scoring.csv --score_ref_labels


