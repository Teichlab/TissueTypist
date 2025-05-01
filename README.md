# TissueTypist
A workflow for training a logistic regression model that leverages both intrinsic transcriptomic signatures and spatial context—incorporating neighboring regions’ profiles and distance to the tissue edge. <br>
Its prediction pipeline is compatible with high-resolution spatial transcriptomics datasets (e.g., Visium HD, Xenium, MERFISH) via a pseudobulk strategy.

## Installation
Creat a conda environment and install with pip
```
conda create --name tissuetypist_env python=3.10
conda activate tissuetypist_env
pip install git+https://github.com/Teichlab/TissueTypist.git
```

## Usage and Documentation
**Please refer to the [demo notebooks](https://github.com/kazukane/TissueTypist/tree/main/notebooks).**
