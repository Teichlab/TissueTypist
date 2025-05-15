# TissueTypist
![Summary of the Workflow](images/tt_workflow.png)
A workflow for training a logistic regression model that leverages both intrinsic transcriptomic signatures and spatial context—incorporating neighboring regions’ profiles and distance to the tissue edge (left box). <br>
Its prediction pipeline is compatible with high-resolution spatial transcriptomics datasets (e.g., Visium HD, Xenium, MERFISH) via a pseudobulk strategy (right box).

## Installation
Creat a conda environment and install with pip
```
conda create --name tissuetypist_env python=3.10
conda activate tissuetypist_env
pip install git+https://github.com/Teichlab/TissueTypist.git
```

## Usage and Documentation
Please refer to the [demo notebooks](https://github.com/kazukane/TissueTypist/tree/main/notebooks) for examples using query datasets of various types and resolutions.

For imaging-based spatial transcriptomics data using targeted gene panels (e.g., Xenium, MERFISH), we recommend training a new model on the panel’s gene set. The demo notebooks include a dedicated training workflow with the built-in Visium reference dataset.
