import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def prepare_dataframe(adata, # need to have XY coordinate in the adata.obsm['spatial']
                      section_col,
                      tissue_col=None,
                      features='all'):
    # get expression data
    if features!='all':
        data = adata[:,features].to_df()
    else:
        data = adata.to_df()
    # coordinates
    data[['x','y']] = adata.obsm['spatial'].copy()
    # section
    data['section'] = adata.obs[section_col].copy()
    # tissue label
    if tissue_col!=None:
        data['tissue'] = adata.obs[tissue_col].copy()
    return data # spot-or-window data in each row

def annotate_edge(data,                  # dataframe contains columns for "expression data","XY coordinates","tissue label" and "section ID"
                  ratio_thresh,          # threshold of ratio of number of "close neighbours" to the value of non-edge spots
                  plot=True              # plotting to check the annotation
                 ): 
    # Initialize new columns
    data['ratio_close_neighbours'] = 0.0
    data['is_edge'] = False
    
    # annotate "edge" data per section
    for section, df in data.groupby('section'):
        coords = df[['x','y']].values
        
        ### Compute cutoff distance to get "close neighbours" ###
        # Detecting nearest‑neighbours for each data point
        nbrs = NearestNeighbors(n_neighbors=6+1).fit(coords) # "+1" since this includes self data
        # Get distances; first column will be the distance to self (0) and (n+1)th column has the distances to the n-th closest data point
        distances, _ = nbrs.kneighbors(coords)
        # Decide cutoff distance to get "close neighbours"
        # Here, taking 95th percentile of the distances to the 6th closest data point. To avoid potential outlier
        # So most of the spots will have more than 6 data points as "close neighbours"
        cutoff = np.percentile(distances[:,6],95)
        
        ### Compute number of "close neighbours", and ratio of the value to the value of non-edge spot ###
        # Count neighbours within that cutoff distance (radius search)
        nbrs_radius = NearestNeighbors(radius=cutoff).fit(coords)
        neighbours = nbrs_radius.radius_neighbors(coords, return_distance=False)
        counts = np.array([len(ids) - 1 for ids in neighbours])  # number of "close neighbours". subtracting self 
        # The 95th percentile value is likely to be a value for a non-edge spot, and at the same time ignoring potential outlier
        # We can calculate the ratio of each data point using the 95th percentile value as reference
        ratios = counts / np.percentile(counts,95)
        
        ### Annotate edge and write back into dataframe ###
        data.loc[df.index, 'ratio_close_neighbours'] = ratios
        data.loc[df.index, 'is_edge'] = ratios<ratio_thresh
        
        if plot:
            df['is_edge'] = ratios < ratio_thresh
            # prepare side-by-side plot
            fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
            # Plot distribution of counts
            axs[0].hist(counts,bins=max(counts)+1)
            axs[0].set_title("Direct neighbour counts")
            axs[0].set_xlabel("count")
            axs[0].set_ylabel("n_spot")
            # Plot interior vs edge
            sns.scatterplot(
                x='x', y='y', hue='is_edge', data=df, s=15,
                palette={False: 'blue', True: 'red'},
                ax=axs[1]
            )
            axs[1].set_title(f"Section {section} — Edge Detection (cutoff={cutoff:.2f})")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
            axs[1].legend(title="Is Edge?")
            # Tight layout & show
            plt.tight_layout()
            plt.show()
    
    return data

def include_neighbours(data, # dataframe contains columns for "expression data","XY coordinates","tissue label" and "section ID"
                       k = 6
                      ):
    # Take expression features
    expression_features = [x for x in data.columns if x not in ['x','y','section','tissue','is_edge','ratio_close_neighbours']]
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
        neighbor_mean = np.zeros_like(X_gene)
        neighbor_max = np.zeros_like(X_gene)
        neighbor_var = np.zeros_like(X_gene)

        for i in range(n_samples_section):
            neighbor_idx = indices[i][1:]  # Exclude the spot itself
            neighbor_values = X_gene[neighbor_idx, :]  # shape: (k_section, n_genes)
            neighbor_mean[i, :] = neighbor_values.mean(axis=0)
            neighbor_max[i, :] = neighbor_values.max(axis=0)
            neighbor_var[i, :] = neighbor_values.var(axis=0)

        # Combine original features with neighbor summary statistics
        # New features are: original, neighbor_mean, neighbor_max, neighbor_var
        X_aug = np.hstack([X_gene, neighbor_mean, neighbor_max, neighbor_var])

        # Create a DataFrame for the augmented features for this section
        # Generate column names for neighbor stats for clarity
        mean_cols = [f"{col}_mean" for col in expression_features]
        max_cols = [f"{col}_max" for col in expression_features]
        var_cols = [f"{col}_var" for col in expression_features]
        all_feature_names = expression_features + mean_cols + max_cols + var_cols

        section_aug_df = pd.DataFrame(X_aug, columns=all_feature_names, index=spot_ids)

        # Add back the 'section', 'x', 'y', 'group', and 'is_edge' columns for later reference or splitting
        section_aug_df['section'] = section
        section_aug_df['x'] = group_df['x'].values
        section_aug_df['y'] = group_df['y'].values
        section_aug_df['group'] = group_df['group'].values
        section_aug_df['is_edge'] = group_df['is_edge'].values

        augmented_data_list.append(section_aug_df)
    
    # Combine all sections into one augmented DataFrame
    augmented_data = pd.concat(augmented_data_list)
    return augmented_data
