import scanpy as sc
import anndata
import pandas as pd
import numpy as np
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
                  edge_feature,
                  neighbour_weight,
                  edge_weight):
    # Build a ColumnTransformer: scale gene/neighbor features normally, and apply custom amplification for edge feature.
    preprocessor = ColumnTransformer([
                ("own_scale", StandardScaler(), own_features),
                ("neigh_scale_weight", Pipeline([
                    ("scale", StandardScaler()),
                    ("weight", utils.AmplifyTransformer(factor=neighbour_weight))
                ]), neighbour_features),
                ("edge_amp", utils.AmplifyTransformer(factor=edge_weight), [edge_feature])
            ])
    # Build the pipeline with the preprocessor and the logistic regression classifier.
    pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs', C=0.01))
            ])
    return pipeline

def train(data,
          own_features,
          neighbour_features,
          edge_feature,
          neighbour_weight,
          edge_weight,
          tissue_col='group',
          save_path=None
         ):
    # build pipeline
    pipeline = _build_pipeline(data,
                              own_features,
                              neighbour_features,
                               edge_feature,
                              neighbour_weight,
                              edge_weight)
    # prepare data
    X_full_df = data[own_features+neighbour_features+[edge_feature]]
    y = data[tissue_col].values
    data[tissue_col] = data[tissue_col].astype('category')
    classes = list(data[tissue_col].cat.categories)
    # Fit the pipeline
    pipeline.fit(X_full_df, y)
    # save the model
    if save_path!=None:
        if save_path.endswith('.joblib'):
            joblib.dump(pipeline,save_path)
        else:
            raise ValueError(f"Invalid save_path: '{save_path}'. File name must end with '.joblib'.")

def cross_validation(data,
                      own_features,
                      neighbour_features,
                     edge_feature,
                      neighbour_weight,
                      edge_weight,
                      tissue_col='group',
                     ):
    # build pipeline
    pipeline = _build_pipeline(data,
                              own_features,
                              neighbour_features,
                               edge_feature,
                              neighbour_weight,
                              edge_weight)
    # prepare data
    X_full_df = data[own_features+neighbour_features+[edge_feature]]
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