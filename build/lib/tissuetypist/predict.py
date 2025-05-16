import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

#gives access to submodules
from . import utils
from . import load_data

# predict with a trained pipeline
def _predict(query_df,pipeline):
    # Get the exact column order the pipeline expects
    cols = pipeline.named_steps["preprocessor"].feature_names_in_
    # Reindex (adds missing cols filled with 0, drops extras)
    X = query_df.reindex(columns=cols, fill_value=0)
    print(f'number of features: {X.shape[1]}')
    # Predict and return
    print(f'predicting...')
    predicted_labels = pipeline.predict(X)
    print(f'done!')
    return predicted_labels

def predict(query_df,
            pipeline_dir=None # if None, use the trained model for a full transcriptome dataset, which is included in this package.
           ):
    if pipeline_dir==None:
        print('Loading the default trained models for a full transcriptome dataset.')
        model_dict = load_data.trained_models_full()
    else:
        # get trained pipelines
        pipeline_list = [x for x in os.listdir(pipeline_dir) if '_pipeline.joblib' in x]
        pipeline_list.sort()
        model_dict={}
        for file in pipeline_list:
            pipeline_name = file.replace('_pipeline.joblib','')
            path_to_pipeline = f"{pipeline_dir}/{file}"
            print('Loading custom models...')
            model_dict[pipeline_name]=joblib.load(path_to_pipeline)
    
    # predict
    for pipeline_name,pipeline in model_dict.items():
        # predict
        print(f'##### {pipeline_name} #####')
        y_predict = _predict(query_df,pipeline)
        # store in the input dataframe
        query_df[f'predicted_labels_{pipeline_name}'] = y_predict.copy()
    
    return query_df

def prediction_to_adata(adata,
                     query_df,
                     weight_neighbour,
                     weight_edge,
                     sliding_window_col=None,
                     ):
    # column which has the prediction results to transfer
    prediction_col = f'predicted_labels_weight2neighbours-{weight_neighbour}_weight2edge-{weight_edge}'
    
    # map if the query data is pseudobulk per sliding window
    if sliding_window_col!=None:
        # mapping dictionary
        mapping_dict = query_df[prediction_col].to_dict()
        # map
        series = adata.obs[sliding_window_col].astype('str').copy()
        adata.obs['tt_prediction'] = series.map(mapping_dict).fillna(series)
    else:
        adata.obs['tt_prediction'] = query_df[prediction_col].reindex(adata.obs_names)

    # check how many NaN
    print(f"{sum(adata.obs['tt_prediction'].isna())} data don't have a predicted result")
    
    return adata