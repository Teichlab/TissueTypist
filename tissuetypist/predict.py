import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

#gives access to submodules
from . import utils

def predict(pipeline_path, query_df):
    # read in pipeline
    pipeline = joblib.load(pipeline_path)
    # Get the exact column order the pipeline expects
    cols = pipeline.named_steps["preprocessor"].feature_names_in_
    # Reindex (adds missing cols filled with 0, drops extras)
    X = query_df.reindex(columns=cols, fill_value=0)
    # Predict and return
    return pipeline.predict(X)
    