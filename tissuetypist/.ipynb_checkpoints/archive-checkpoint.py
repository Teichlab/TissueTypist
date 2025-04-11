import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple

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

# a helper function for "prepare_dataframe"
def _include_neighbours_archive(data, # dataframe contains columns for "expression data","XY coordinates","tissue label" and "section ID"
                       k = 6
                      ):
    # Take expression features
    expression_features = [x for x in data.columns if x not in ['x','y','section','tissue']]
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
        if 'tissue' in group_df.columns:
            section_aug_df['tissue'] = group_df['tissue'].values
        
        # append
        augmented_data_list.append(section_aug_df)
    
    # Combine all sections into one augmented DataFrame
    augmented_data = pd.concat(augmented_data_list)
    return augmented_data


# 9 Apr 2025
def annotate_edge(data, # dataframe obtained by running "prepare_dataframe"
                  tile_type='hex', # 'hex' or 'sliding_window'
                  remove_technical_edge=True,
                  distance_to_edge=True,
                 plot=True):
    # check 'title_type' argument
    if tile_type not in ('hex','sliding_window'):
        raise ValueError("Invalid title_type value. Expected 'hex' or 'sliding_window'.")
    
    # Initialize new columns
    data['n_neighbours'] = np.nan
    data['is_edge'] = False
    data['distance_to_edge'] = np.nan
    
    # annotate "edge" data per section
    for section, df in data.groupby('section'):
        coords = df[['x','y']].values
        if tile_type=='hex':
            nbrs = NearestNeighbors(n_neighbors=6+1).fit(coords) # "+1" since this includes self data
        else:
            nbrs = NearestNeighbors(n_neighbors=4+1).fit(coords) # "+1" since this includes self data
        # Get distances; first column will be the distance to self (0) and (n+1)th column has the distances to the n-th closest data point
        distances, _ = nbrs.kneighbors(coords)
        # Get reference distance: minimum distance between data points
        ref_distance = np.min(distances[:,1])
        # based on the ref_distance, count number of direct neighbours
        # Set ratio (to ref_distance) threshold as 1.1. To accept minor variation of the distances to direct neighbours.
        n_neighbour = ((distances[:,1:]/ref_distance)<1.1).sum(axis=1)
        # Annotate edge and write back into dataframe
        data.loc[df.index, 'n_neighbours'] = n_neighbour
        if tile_type=='hex':
            data.loc[df.index, 'is_edge'] = n_neighbour <= 4
            df['is_edge'] = n_neighbour <= 4 # for plotting
        else:
            data.loc[df.index, 'is_edge'] = n_neighbour <= 3
            df['is_edge'] = n_neighbour <= 3 # for plotting

        # remove technical edge
        if remove_technical_edge:
            technical_edge_ids = _find_technical_edge(df)
            data.loc[technical_edge_ids, 'is_edge'] = False
            df.loc[technical_edge_ids, 'is_edge'] = False # for plotting
            
        # calculate distance to edge
        if distance_to_edge:
            df = _distance_to_edge(df,norm=True,log1p=True)
            data.loc[df.index,'distance_to_edge'] = df['distance_to_edge']
            
        # plot for checking
        if plot:
            if distance_to_edge:
                sns.scatterplot(
                    x='x',y='y',hue='distance_to_edge',data=df,
                    palette='rainbow_r',s=10
               )
                plt.legend(title='distance to edge\n(normalised)',bbox_to_anchor=(1, 1))
            else:
                sns.scatterplot(
                    x='x', y='y', hue='is_edge', data=df, 
                    palette={False: 'blue', True: 'red'},s=10
                )
                plt.legend(title="Is Edge?",bbox_to_anchor=(1, 1))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(section)
            plt.show()
    
    return data


# this is a helper function for "find_technical_edge"
def _find_arithmetic_segments(series: pd.Series):
    # Convert the Series values to a numpy array for easy slicing.
    values = series.values
    # Get the element IDs (index) as a numpy array.
    element_ids = series.index.to_numpy()
    
    arithmetic_segments = []  # Will hold tuples (segment_indices, common_diff)
    n = len(values)
    i = 0
    
    while i < n:
        # Start a new subsequence from index i; must have at least 2 elements.
        if i == n - 1:
            break  # no pair available
        # Determine the difference for this potential arithmetic sequence.
        diff = values[i+1] - values[i]
        j = i + 1
        # Extend the sequence as long as the difference is maintained.
        while j < n - 1 and (values[j+1] - values[j] == diff):
            j += 1
        # Only consider subsequences of length at least 2.
        if j - i + 1 >= 2:
            # Save the corresponding element IDs from the index.
            seg_ids = element_ids[i:j+1]
            # arithmetic_segments.append((seg_ids, diff))
            arithmetic_segments.append(seg_ids)
        i = j + 1
  
    # get segments whose length are 5 or larger
    result = []
    for seg in arithmetic_segments:
        if len(seg)>=5:
            result = result + list(seg)
    return result

    '''
    # If no arithmetic segments were found, return an empty list.
    if not arithmetic_segments:
        return []
    
    # Find the smallest common difference among the segments.
    min_diff = min(diff for seg, diff in arithmetic_segments)
    
    # Filter segments that have the smallest common difference.
    result = [seg for seg, diff in arithmetic_segments if diff == min_diff]
    
    return result
    '''

def sliding_window2(
    adata: AnnData | SpatialData,
    library_key: str | None = None,
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool = False,
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    %(library_key)s
    sliding_window_key: str
        Base name for sliding window columns.
    %(spatial_key)s
    drop_partial_windows: bool
        If True, drop windows that are smaller than the window size at the borders.
    copy: bool
        If True, return the result, otherwise save it to the adata object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """

    if isinstance(adata, SpatialData):
        adata = adata.table

    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    # extract coordinates of observations
    coord_columns = ("globalX", "globalY")
    x_col, y_col = coord_columns
    if spatial_key in adata.obsm:
        coords = pd.DataFrame(
            adata.obsm[spatial_key][:, :2],
            index=adata.obs.index,
            columns=[x_col, y_col],
        )
    else:
        raise ValueError(
            f"Coordinates not found. Specify a suitable `spatial_key` in `adata.obsm`."
        )

    # infer window size if not provided
    window_size = 50

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")

    for lib in libraries:
        if lib is not None:
            lib_mask = adata.obs[library_key] == lib
            lib_coords = coords.loc[lib_mask]
        else:
            lib_mask = np.ones(len(adata), dtype=bool)
            lib_coords = coords

        min_x, max_x = lib_coords[x_col].min(), lib_coords[x_col].max()
        min_y, max_y = lib_coords[y_col].min(), lib_coords[y_col].max()

        # precalculate windows
        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            drop_partial_windows=drop_partial_windows,
        )

        lib_key = f"{lib}_" if lib is not None else ""

        # assign observations to windows
        for idx, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]

            mask = (
                (lib_coords[x_col] >= x_start)
                & (lib_coords[x_col] <= x_end)
                & (lib_coords[y_col] >= y_start)
                & (lib_coords[y_col] <= y_end)
            )
            obs_indices = lib_coords.index[mask]
            sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"

            ############ added by KK ############
            # to get coordinates of each window
            # center point
            sliding_window_df.loc[obs_indices, 'window_col'] = x_start+(window_size/2)
            sliding_window_df.loc[obs_indices, 'window_row'] = y_start+(window_size/2)
            #####################################

    # create categorical variable for ordered windows
    sliding_window_df[sliding_window_key] = pd.Categorical(
        sliding_window_df[sliding_window_key],
        ordered=True,
        categories=sorted(
            sliding_window_df[sliding_window_key].unique(),
            key=lambda x: int(x.split("_")[-1]),
        ),
    )

    sliding_window_df[x_col] = coords[x_col]
    sliding_window_df[y_col] = coords[y_col]

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)

##### 10Apr2025 #####
def preprocess_data(adata,
                  section_col,
                  coord_columns: Optional[Tuple[str, str]], # if None, need to have XY coordinate in the adata.obsm['spatial']
                  tissue_col=None,
                  gene_panel=None,
                  tile_type='square', # 'hexagon' or 'square'
                  remove_technical_edge=True,
                  plot=False
                 ):
    ### Pre-process
    # check whether the gene count is raw count
    try:
        if not np.all(adata.X.data % 1 == 0):
            warnings.warn("Gene counts contain non-integer values!")
        else:
            print("Gene counts are all integer values.")
    except: # in case .X is numpy.ndarray
        arr = adata.X.ravel()[:1000]
        if not np.all(arr % 1 == 0):
            warnings.warn("Gene counts contain non-integer values!")
        else:
            print("Gene counts are all integer values.")
    # subset feature based on the gene_panel of the query data
    # subsetting needs to be before log-normalisation
    if gene_panel!=None:
        shared = list(set(adata.var_names).intersection(gene_panel))
        if len(shared)==0:
            raise ValueError("Error: no shared genes between adata and gene_panel!")
        adata = adata[:,shared]
    # filter & log-normalise
    print("Filtering genes & log-normalising...")
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ### Get expression data (own features)
    # will base on key_tissue_genes (HVGs+DEGs across tissues of the training dataset)
    key_tissue_genes = load_data.key_tissue_genes()
    shared = list(set(adata.var_names).intersection(key_tissue_genes))
    if len(shared)==0:
        raise ValueError("Error: no shared genes between adata and key_tissue_genes!")
    else:
        print(f'{len(shared)} genes will be used')
    data = adata[:,shared].to_df()
    data.columns = [f'{x}_own' for x in data.columns]
    
    ### Get section ID
    data['section'] = adata.obs[section_col].copy()
    
    ### Get coordinates
    if coord_columns==None:
        data[['x','y']] = adata.obsm['spatial'].copy()
    else:
        data[['x','y']] = adata.obs[[coord_columns[0],coord_columns[1]]]
    
    ### Get tissue label
    if tissue_col!=None:
        data['tissue'] = adata.obs[tissue_col].copy()
    
    ### Include neighbour data
    data = _include_neighbours(data,k=6)
    
    ### Annotate edge
    print("Calculating distance from tissue edge...")
    data = annotate_edge.annotate_edge(data, # dataframe obtained by running "prepare_dataframe"
                                      tile_type=tile_type, # 'hex' or 'sliding_window'
                                      remove_technical_edge=remove_technical_edge,
                                     plot=plot)
    return data # spot-or-window data in each row

def preprocess_reference_data(gene_panel,
                           plot=False):
    # read in reference adata
    adata = load_data.reference_adata()
    # prepare
    data = prepare_data(adata,
                      section_col='sample',
                      coord_columns=('array_col','array_row'),
                      tissue_col='annotation_final_mod',
                      gene_panel=gene_panel,
                      tile_type='square',
                      remove_technical_edge=True,
                      plot=plot
                     )
    # remove tissue-unassigned spots
    data = data[data['tissue']!='unassigned']
    return data