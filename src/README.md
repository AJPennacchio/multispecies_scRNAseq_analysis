Note that you must have SATURN repo cloned to your workspace. https://github.com/snap-stanford/SATURN. Basic setup instructions can be found under "Setting up SATURN" section on home page. 

The basic pipeline is as follows:

1 --> Preprocessing for count matrices (h5ad files). Basic filtering, normalizing, PCA, neighbors, leiden, UMAP computation. Note that HV genes should not be extracted and count data should not be log transformed. Also note that scanpy does not have a method to read in cellbender processed matrices (as of 8/8/23). A workaround is to use the anndata_from_h5() method which can be found in the preprocessing notebook of the dataset-specific pipeline in this repo.

Input: Paths to unprocessed h5ad file and path to save processed h5ad file
Output: adata object, saves it to results file

1.5 --> Generate protein embeddings, detailed instructions here: https://github.com/snap-stanford/SATURN/tree/main/protein_embeddings
	Note that you must clone ESM repo to generate protein embeddings. 

2 --> Create CSV file containing necessary info for SATURN. Update species names, paths for processed h5ad files and protein embedding files, and csv file location prior to running. 

3 --> Wrapper script for running SATURN. 

Sample Input: bash 3_run_SATURN.sh /home/apennacchio/SATURN/train-saturn.py /home/apennacchio/pcsct/saturn/s10_z23.csv leiden lab

	$1 --> path to train-saturn.py file in SATURN repo

	$2 --> path to csv, generated in step 2

	$3 --> column name in processed h5ad files to guide SATURN's integration (leiden or cell type labels)

	$4 --> reference labels (optional)

Output: directory with various SATURN output files

4 --> Visualize SATURN's results.

Input: path to SATURN output file, should  look something like "test256_data_SAMPLE_NAMES_org_saturn_seed_0.h5ad" 
Output: UMAPs

5 --> Uses XGB classifier to transfer labels to gauge SATURN's performance. Uses ref_label column in adata. May need to update number of PCs used in function to encompass about 90% of variation.  

Input: SATURN output file (same as in step 4), train species name (exactly as appears in species column in adata), test species name (exactly as appears in species column in adata).
Output: Label transfer accuracy, from species1 to species2
