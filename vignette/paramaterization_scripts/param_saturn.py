import numpy as np
import pandas as pd
import scanpy as sc
import tables
import scipy.sparse as sp
import anndata
from typing import Dict, Optional
from matplotlib.pyplot import rc_context
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import subprocess
import shlex
import os

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

#No scanpy function to load in cellranger matrix

def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> 'anndata.AnnData':
    """Load an output h5 file into an AnnData object for downstream work.

    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.

    Returns:
        adata: The anndata object, populated with inferred latent variables
            and metadata.

    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]

    return adata


def dict_from_h5(file: str) -> Dict[str, np.ndarray]:
    """Read in everything from an h5 file and put into a dictionary."""
    d = {}
    with tables.open_file(file) as f:
        # read in everything
        for array in f.walk_nodes("/", "Array"):
            d[array.name] = array.read()
    return d


def _fill_adata_slots_automatically(adata, d):
    """Add other information to the adata object in the appropriate slot."""

    for key, value in d.items():
        try:
            if value is None:
                continue
            value = np.asarray(value)
            if len(value.shape) == 0:
                adata.uns[key] = value
            elif value.shape[0] == adata.shape[0]:
                if (len(value.shape) < 2) or (value.shape[1] < 2):
                    adata.obs[key] = value
                else:
                    adata.obsm[key] = value
            elif value.shape[0] == adata.shape[1]:
                if value.dtype.name.startswith('bytes'):
                    adata.var[key] = value.astype(str)
                else:
                    adata.var[key] = value
            else:
                adata.uns[key] = value
        except Exception:
            print('Unable to load data into AnnData: ', key, value, type(value))


def preprocess(sample, res):
    # read adata and specify results file
    results_file = '/home/apennacchio/pcsct/data/preprocessed/' + sample + '.h5ad'
    adata = anndata_from_h5(file='/home/apennacchio/pcsct/data/cellbender/' + sample + '/' + sample + '_filtered.h5')

    # remove gene: prefix from all gene names
    new_gene_names_in_correct_order = [id[5:] if id[0:5] == 'gene:' else id for id in adata.var_names]
    adata.var_names = new_gene_names_in_correct_order

    adata.var_names_make_unique()

    # filtering by UMI, min genes, min cells
    # sc.pl.highest_expr_genes(adata, n_top=20, )

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    if (sample[0:3] == 'sbi'):
        print("Sorghum UMI cutoffs")
        sc.pp.filter_cells(adata, min_counts=1200)
        sc.pp.filter_cells(adata, max_counts=80000)
    elif (sample[0:3] == 'zma'):
        print("Maize UMI cutoffs")
        sc.pp.filter_cells(adata, min_counts=2000)
        sc.pp.filter_cells(adata, max_counts=200000)

    # normalize + log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["LogTransform"] = adata.X.copy()
    ## sc.pp.log1p(adata, layer="LogTransform")  # note, SATURN requires count data##

    # filter highly variable genes -  SATURN doesn't like this
   ## sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                ##                layer="LogTransform")  # these are default values##
    # adata.raw = adata #save raw data before processing values and further filtering
   ## adata = adata[:, adata.var.highly_variable]  # filter highly variable##
    # sc.pp.scale(adata, max_value=10, layer="LogTransform") #scale each gene to unit variance##
   ## del adata.layers["LogTransform"]

    # PCA, Neighborhood, Leiden, UMAP
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)  # , n_neighbors=20, n_pcs=40)
    sc.tl.leiden(adata, resolution=res)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='leiden')
    # adata.write(results_file)

    return adata


def extract_CT(adata, sample):
    results_file = '/home/apennacchio/pcsct/data/preprocessed/' + sample + '.h5ad'
    df = pd.read_csv('/home/apennacchio/pcsct/R_scripts/labels.csv')
    sources = []

    # Remove cell subtypes
    df['lab'].replace(['Cortex_1', 'Cortex_2', 'Cortex_3', 'Cortex_4'], 'Cortex', inplace=True)
    df['lab'].replace(['Epidermis_1', 'Epidermis_2', 'Epidermis_3'], 'Epidermis', inplace=True)
    df['lab'].replace(['Young_Stele_2'], 'YoungStele', inplace=True)
    df['lab'].replace(['Stele_2', 'Stele_1'], 'Stele', inplace=True)
    df['lab'].replace(['Young_Phloem'], 'YoungPhloem', inplace=True)
    df['lab'].replace(['G2_M'], 'G2', inplace=True)

    if (sample[0:3] == 'sbi'):
        df = df.loc[df['species'].str.contains("Sorghum")]
        sources = ["Sorghum_Cell_", "Sorghum_Nucl1_", "Sorghum_Nucl2_", "Sorghum_Nucl3_"]
        # print(df)
    elif (sample[0:3] == 'zma'):
        df = df.loc[df['species'].str.contains("Maize")]
        sources = ["Maize_Cell1_", "Maize_Cell2_", "Maize_Cell3_", "Maize_Cell4_", "Maize_Nucl1_", "Maize_Nucl2_",
                   "Maize_Nucl3_", "Maize_Nucl4_"]
        # print(df)

    df = df.drop(["species"], axis=1)
    df["source"] = ""

    for i in sources:
        mask = df['barcodes'].str.contains(i)
        df['barcodes'] = df['barcodes'].str.replace(i, "")
        df.loc[mask, 'source'] = i

    df = df.set_index("barcodes")

    intersection = np.intersect1d(adata.obs.index, df.index)
    df = df.loc[intersection]
    df = df[~df.index.duplicated()]
    adata = adata[intersection]
    adata.obs = adata.obs.join(df, how='inner')
    # adata.obs = adata.obs.join(df, how='left')
    # adata.obs = adata.obs.fillna("Unlabeled")
    adata.write(results_file)

    return adata


def ingest_samples(arr):
    ref = arr[0]
    dat = arr[1]
    ref.obs['Sample'] = '2'
    dat.obs['Sample'] = '3'
    var_names = ref.var_names.intersection(dat.var_names)
    dat = dat[:, var_names]
    ref = ref[:, var_names]
    sc.tl.ingest(dat, ref, obs='leiden')
    ref = extract_CT(ref, "zmays_root2")
    dat = extract_CT(dat, "zmays_root3")
    merged = ref.concatenate(dat)
    merged.write('/home/apennacchio/pcsct/data/preprocessed/integrated_zmays23.h5ad')

    return merged

def make_csv(res, mg):
    #create csv containing h5 files, species, and protein embeddings as per SATURN's input requirements

    df = pd.DataFrame(columns=["path", "species", "embedding_path"])

    #change paths for each run

    #species
    df["species"] = ["sorghum", "zmays"]

    #h5 file
    df["path"] = ["/home/apennacchio/pcsct/data/preprocessed/sbicolor_root10.h5ad",
        "/home/apennacchio/pcsct/data/preprocessed/integrated_zmays23.h5ad",
        ]

    #protein embeddings
    sorghum_embedding_path = "/home/apennacchio/SATURN/protein_embeddings/proteome/embeddings/Sorghum_bicolor_NCBIv3.gene_symbol_to_embedding_ESM1b.pt"
    zmays_embedding_path = "/home/apennacchio/SATURN/protein_embeddings/proteome/embeddings/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.gene_symbol_to_embedding_ESM1b.pt"
    df["embedding_path"] = [sorghum_embedding_path, zmays_embedding_path]

    #save csv location
    csv_path = "/home/apennacchio/pcsct/saturn/s10_z23_" + str(res) + "-" + str(mg) + ".csv"
    df.to_csv("/home/apennacchio/pcsct/saturn/s10_z23_" + str(res) + "-" + str(mg) + ".csv",index=False)
    return os.path.basename(csv_path)


def transfer_labels(folder):
    adata = sc.read(
        "/home/apennacchio/pcsct/saturn/out/" + folder + "/test256_data_sbicolor_root10_integrated_zmays23_org_saturn_seed_0.h5ad")
    sc.pp.pca(adata, n_comps=14)
    pc_df = pd.DataFrame(adata.obsm['X_pca'])
    pc_df = pc_df.set_index(adata.obs.index)
    final_df = pd.concat([pc_df, adata.obs['ref_labels'], adata.obs['species']], axis = 1)
    final_df = final_df.drop(final_df[final_df['ref_labels'] == 'Unlabeled'].index)
    y = final_df["ref_labels"]
    labelencoder = LabelEncoder()
    final_Y = labelencoder.fit_transform(y)
    final_df["category"] = final_Y
    df_s = final_df[final_df['species'] == 'sorghum']
    df_z = final_df[final_df['species'] == 'zmays']
    df_s = df_s.drop(labels=["species"], axis=1)
    df_z = df_z.drop(labels=["species"], axis=1)
    X_train = df_s.drop(labels=["ref_labels", "category"], axis=1)
    Y_train = df_s["category"]
    X_test = df_z.drop(labels=["ref_labels", "category"], axis=1)
    Y_test = df_z["category"]
    #Gaussian test
    # gnb = GaussianNB()
    # y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != y_pred).sum()))
    # print(y_pred.size)
    # gauss = 1 - ((Y_test != y_pred).sum() / y_pred.size)
    #XGBoost test
    bst = XGBClassifier(learning_rate =0.1,
     n_estimators=1000,
     max_depth=4,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     seed=27)
    bst.fit(X_train, Y_train)
    preds = bst.predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != preds).sum()))
    xgbo = 1 - ((Y_test != preds).sum() / preds.size)
    
    return xgbo

def main():
    results = pd.DataFrame(columns=['cluster_res', 'num_HV', 'label_transfer'])
    resolutions = [0.75, 1.5, 2.25, 3, 3.75, 4.5]
    num_HV = [5000, 6000, 7000, 8000, 9000, 10000]
    for res in resolutions:
      samples = []
      for i in ["sbicolor_root10", "zmays_root2", "zmays_root3"]:
         samples.append(preprocess(i, res))
      ingest = [samples[1], samples[2]]
      merged = ingest_samples(ingest)
      sbi = extract_CT(samples[0], "sbicolor_root10")

      for HV in num_HV:    
           run =  make_csv(res, HV)
           subprocess.call(shlex.split(f"./sat.sh {res} {HV} {run}"))
           new_folder = "./out/" + str(res) + "_" + str(HV) + "num_HV"
           os.rename("./out/saturn_results", new_folder )
           accuracy = transfer_labels(str(res) + "_" + str(HV) + "num_HV")       
           results.loc[len(results.index)] = [res, HV, accuracy]
    results.to_csv("summary_num_HV.csv", index=False)

if __name__=="__main__":
    main()
