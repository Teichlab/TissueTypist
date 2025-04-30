import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple
import warnings

from . import load_data
from . import annotate_edge
from . import sliding_window

# a helper function for "prepare_dataframe"
def _include_neighbours(data, # dataframe contains columns for "expression data","XY coordinates","tissue label" and "section ID"
                       k = 6
                      ):
    # Take expression features
    expression_features = [x for x in data.columns if '_own' in x]
    # Process each section separately
    augmented_data_list=[]
    for section, group_df in data.groupby('section'):
        # Extract spatial coordinates and gene expression for this section
        spot_ids = group_df.index
        coords = group_df[['x', 'y']].values
        X_gene = group_df[expression_features].values  # shape: (n_samples_section, n_genes)
        n_samples_section, n_genes = X_gene.shape

        # Set number of neighbors (excluding self); adjust if the section has too few spots
        k_section = k if n_samples_section > k + 1 else max(n_samples_section - 1, 1)

        # Determine neighbors using KNN within this section
        nbrs = NearestNeighbors(n_neighbors=k_section + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Compute summary statistics for each spot from its neighbors
        neighbor_max = np.zeros_like(X_gene)
        for i in range(n_samples_section):
            neighbor_idx = indices[i][1:]  # Exclude the spot itself
            neighbor_values = X_gene[neighbor_idx, :]  # shape: (k_section, n_genes)
            neighbor_max[i, :] = neighbor_values.max(axis=0)

        # Combine original features with neighbor summary statistics
        # New features are: original, neighbor_mean, neighbor_max, neighbor_var
        X_aug = np.hstack([X_gene, neighbor_max])

        # Create a DataFrame for the augmented features for this section
        # Generate column names for neighbor stats for clarity
        max_cols = [col.replace('_own','_neighbour-max') for col in expression_features]
        all_feature_names = expression_features + max_cols
        section_aug_df = pd.DataFrame(X_aug, columns=all_feature_names, index=spot_ids)

        # Add back the 'section', 'x', 'y', 'group', and 'is_edge' columns for later reference or splitting
        section_aug_df['section'] = section
        section_aug_df['x'] = group_df['x'].values
        section_aug_df['y'] = group_df['y'].values
        if 'tissue' in group_df.columns:
            section_aug_df['tissue'] = group_df['tissue'].values
        
        # append
        augmented_data_list.append(section_aug_df)
    
    # Combine all sections into one augmented DataFrame
    augmented_data = pd.concat(augmented_data_list)
    return augmented_data

def preprocess(adata,
                  section_col,
                  coord_columns: Optional[Tuple[str, str]], # if None, need to have XY coordinate in the adata.obsm['spatial']
               pseudobulk=False,
               pseudobulk_window_size=None,
                  tile_type='square', # 'hexagon' or 'square'
                  remove_technical_edge=True,
                  plot=False
                 ):
    # this is the function to making pseudobulk, prepare own and neighbour gene expression data, and calculate distance to edge.
    
    ### Pseudobulk per sliding_window, if requires
    if pseudobulk:
        print("Making pseudobulk per sliding_window...")
        bdata = sliding_window.sliding_window_psudobulk(
            adata=adata,
            section_col=section_col,
            window_size=pseudobulk_window_size,
                coord_columns=coord_columns,
                log_normalise=True
            )
        # update "coord_columns"
        coord_columns = ('window_col','window_row')
    else:
        bdata = adata.copy()
    
    ### Get expression data (own features)
    print("Preparing expression data...")
    # will base on key_tissue_genes (HVGs+DEGs across tissues of the training dataset)
    key_tissue_genes = load_data.key_tissue_genes()
    shared = list(set(bdata.var_names).intersection(key_tissue_genes))
    if len(shared)==0:
        raise ValueError("Error: no shared genes between adata and key_tissue_genes!")
    else:
        print(f'{len(shared)} genes will be used')
    data = bdata[:,shared].to_df()
    data.columns = [f'{x}_own' for x in data.columns]
    
    ### Get section ID
    data['section'] = bdata.obs[section_col].copy()
    
    ### Get coordinates
    if coord_columns==None:
        data[['x','y']] = bdata.obsm['spatial'].copy()
    else:
        data[['x','y']] = bdata.obs[[coord_columns[0],coord_columns[1]]]
    
    ### Include neighbour data
    data = _include_neighbours(data,k=6)
    
    ### Annotate edge
    print("Calculating distance from tissue edge...")
    data = annotate_edge.annotate_edge(data, # dataframe obtained by running "prepare_dataframe"
                                      tile_type=tile_type, # 'hex' or 'sliding_window'
                                      remove_technical_edge=remove_technical_edge,
                                     plot=plot)
    return data # spot-or-window data in each row

def preprocess_builtin_reference(gene_panel,
                           plot=False):
    ### read in reference adata
    adata = load_data.reference_adata()

    ### subset feature based on the gene_panel of the query data
    # subsetting needs to be before log-normalisation
    if gene_panel!=None:
        shared = list(set(adata.var_names).intersection(gene_panel))
        if len(shared)==0:
            raise ValueError("Error: no shared genes between adata and gene_panel!")
        adata = adata[:,shared]
        
    ### filter & log-normalise
    print("log-normalising...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    ### preprocess
    data = preprocess(adata,
                      section_col='sample',
                      coord_columns=('array_col','array_row'),
                      tile_type='square',
                      remove_technical_edge=True,
                      plot=plot
                     )
    
    ### Get tissue label
    data.loc[adata.obs_names,'tissue'] = adata.obs['annotation_final_mod'].copy()
    
    ### remove tissue-unassigned spots
    data = data[data['tissue']!='unassigned']
    
    return data


