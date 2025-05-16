import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.base import BaseEstimator, TransformerMixin

class AmplifyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.factor
    

# refer to https://github.com/scverse/scanpy/issues/181#issuecomment-534867254
def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = adata.var[idx]
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out

# DE test with mean expression per group
def rank_genes_groups_with_mean(adata, groupby, groups='all', reference='rest', 
                                method=None, corr_method='benjamini-hochberg',
                                group_for_result=None,
                                pval_cutoff=0.05,
                                log2fc_min=None, log2fc_max=None,
                                **kwds_to_scanpy_rank_test):
    
    # rank test
    # https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html
    sc.tl.rank_genes_groups(adata=adata, 
                            groupby=groupby,
                            groups=groups, reference=reference,
                            method=method, corr_method=corr_method,
                           **kwds_to_scanpy_rank_test)
    
    # get rank test result
    # https://scanpy.readthedocs.io/en/stable/generated/scanpy.get.rank_genes_groups_df.html
    res = sc.get.rank_genes_groups_df(adata, group=group_for_result, 
                                      pval_cutoff=pval_cutoff,
                                     log2fc_min=log2fc_min, log2fc_max=log2fc_max)
    if (groups!='all')&(len(groups)==1):
        res['group']=groups[0]
    
    # calculate mean value per group
    group_mean = grouped_obs_mean(adata, group_key=groupby)
    ## remove index title
    group_mean.index.name = None
    
    # add mean gene expression
    group_mean = group_mean.reset_index().melt(id_vars=['index'])
    group_mean.rename(columns={'index':'names','variable':'group','value':'group_mean'},
                      inplace=True)
    res = res.merge(group_mean, how='left', on=['group','names'])
    
    return res

# function to save list as plain text file
def list2txt(l, path):
    with open(path, 'w') as file:
        for item in l:
            file.write(f"{item}\n")
# function to read text file as list
def txt2list(path):
    with open(path, 'r') as file:
        l = [line.strip() for line in file.readlines()]
    return l