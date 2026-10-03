## Create Conda Environment
1. Download and install [miniconda](https://www.anaconda.com/download) (NOTE: No need to sign up anything!)
2. Create a new environment for the project
```shell
$ conda create --name sparsemap python=3.13
$ conda activate sparsemap
```
3. install packages
```shell
$ conda install -c conda-forge ipython ipykernel nbformat sphinx sphinx-rtd-theme myst-parser matplotlib opencv pandas multiprocess h5py scipy scikit-image scikit-learn networkx
```

## Install `src` Folder (Source Codes)
```shell
$ cd /dir_to_project_folder
$ conda activate sparsemap
$ pip install -e .
```

## Configure `src/metadata.py`
1. Modify `bigdata_dir` to load your dataset

## Run Codes
Run `scripts/*.ipynb`  in the following series:
1. `sparsemap_1,2,3,4,5`
2. `fig_1,2,3`

## View Documentation (under construction)
1. Open the html file in `sparsemap/docs/build/html/index.html`