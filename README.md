# DC3-Reproduction
Replication study of "Data-centric framework for crystal structure identification in atomistic simulations using machine learning" by Chung et al. (2022)

## Tasks
- [x] Synthetic crystal structure generation
- [x] Feature vector generation
    - [ ] Research into faster parameterizations
- [x] ML dataset generation / t-SNE analysis
- [ ] Outlier detection model
- [x] ML model for crystal structure identification
    - [ ] ML training on complex datasets
- [ ] MD simulation and model testing
    - [ ] LAMMPS molecular simulations and visualizations
    - [ ] Model testing performance over different thermal distributions

## Components
- `lattice`: Generate synthetic crystal structures with thermal noise, loading from OVITO/numpy
- `features`: Extract RSF and SOP features from crystal structures
- `ml_dataset`: Pytorch dataset from feature vectors
- `ml`: Pytorch training scripts

## Requirements
- `ovito`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `torch`, `tqdm`
- `conda env create -n dc3 numpy scipy matplotlib scikit-learn`, `conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3`
- Requires `python >= 3.12`

## Train
`python3 -m ml.train`

## Contributors
Ethan Cardenas   
Jieruei Chang   
Alexander Liang   
Fiona Lu   
Dan Xiao