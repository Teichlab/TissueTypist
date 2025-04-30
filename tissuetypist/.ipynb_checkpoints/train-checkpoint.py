import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, make_scorer
import joblib

#gives access to submodules
from . import utils
    
def _build_pipeline(data,
                  own_features,
                  neighbour_features,
                  neighbour_weight,
                  edge_weight):
    # Build a ColumnTransformer: scale gene/neighbor features normally, and apply custom amplification for edge feature.
    preprocessor = ColumnTransformer([
                ("own_scale", StandardScaler(), own_features),
                ("neigh_scale_weight", Pipeline([
                    ("scale", StandardScaler()),
                    ("weight", utils.AmplifyTransformer(factor=neighbour_weight))
                ]), neighbour_features),
                ("edge_amp", utils.AmplifyTransformer(factor=edge_weight), ['distance_to_edge'])
            ])
    # Build the pipeline with the preprocessor and the logistic regression classifier.
    pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=500, solver='lbfgs', C=0.01))
            ])
    return pipeline

# train for a condition
def _train(
          data,
          own_features,
          neighbour_features,
          neighbour_weight,
          edge_weight,
          tissue_col='group',
          save_path_pipeline=None
         ):
    # build pipeline
    pipeline = _build_pipeline(data,
                              own_features,
                              neighbour_features,
                              neighbour_weight,
                              edge_weight)
    # prepare data
    X_full_df = data[own_features+neighbour_features+['distance_to_edge']]
    y = data[tissue_col].values
    data[tissue_col] = data[tissue_col].astype('category')
    classes = list(data[tissue_col].cat.categories)
    # Fit the pipeline
    pipeline.fit(X_full_df, y)
    # save the model
    if save_path_pipeline!=None:
        joblib.dump(pipeline,save_path_pipeline)

# main train function
# train with three conditions: without spatial-info, neighbour-weight-0.3 and neighbour-weight-1
def train(data,
          tissue_col='tissue',
          save_dir=None
         ):
    
    # prepare the directory to save trained models
    if save_dir!=None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print("Warning: This will overwrite the existing models.")
            
    # prepare features
    own_features = [x for x in data.columns if '_own' in x]
    neighbour_features = [x for x in data.columns if '_neighbour-max' in x]
    
    weight_pair_list = [(0.0, 0),
                        (0.3, 5),
                        (1.0, 5)]
    # train per condition
    for neighbour_weight,edge_weight in weight_pair_list:
        print(f"Training...: weight to neighbour spots = {neighbour_weight}, weight to edge = {edge_weight}")
        
        if save_dir!=None:
            save_path_pipeline = f"{save_dir}/weight2neighbours-{neighbour_weight}_weight2edge-{edge_weight}_pipeline.joblib"
        else:
            save_path_pipeline = None

        _train(data,
               own_features=own_features,
               neighbour_features=neighbour_features,
               neighbour_weight=neighbour_weight,
               edge_weight=edge_weight,
               tissue_col=tissue_col,
               save_path_pipeline=save_path_pipeline
                 )
        
def cross_validation(data,
                      own_features,
                      neighbour_features,
                      neighbour_weight,
                      edge_weight,
                      tissue_col='group',
                     ):
    # build pipeline
    pipeline = _build_pipeline(data,
                              own_features,
                              neighbour_features,
                              neighbour_weight,
                              edge_weight)
    # prepare data
    X_full_df = data[own_features+neighbour_features+['distance_to_edge']]
    y = data[tissue_col].values
    data[tissue_col] = data[tissue_col].astype('category')
    classes = list(data[tissue_col].cat.categories)
    # Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro",
            }
    scoring = {
                **scoring,  # keep your existing metrics
                **{f"f1_class_{cls}": make_scorer(f1_score, labels=[cls], average="macro") 
                   for cls in classes}
            }
    cv = cross_validate(pipeline, X_full_df, y, cv=skf, scoring=scoring, return_train_score=False)
    # Build a per‑fold DataFrame for this factor
    df_folds = pd.DataFrame({
        "weight_to_neighbours": neighbour_weight,
        "weight_to_edge": edge_weight,
        "fold": np.arange(len(cv["test_accuracy"])) + 1,
        "accuracy": cv["test_accuracy"],
        "balanced_accuracy": cv["test_balanced_accuracy"],
        "f1_macro": cv["test_f1_macro"]
    })
    # f1 score dataframe
    f1_score_class_df = pd.DataFrame({k: cv[f"test_{k}"] for k in scoring if k.startswith("f1_class_")})
    f1_score_class_df["weight_to_neighbours"] = neighbour_weight
    f1_score_class_df["weight_to_edge"] = edge_weight
    
    return df_folds, f1_score_class_df