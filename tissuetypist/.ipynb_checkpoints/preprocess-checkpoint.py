import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple
import warnings

from itertools import product
from scanpy import logging as logg
from squidpy.gr._utils import _save_data

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from . import load_data

# a helper function for "_sliding_window"
# https://github.com/scverse/squidpy/blob/bdd989983cd24aaa46bbf4751f1e904fe10ecc44/src/squidpy/tl/_sliding_window.py#L178
def _calculate_window_corners(
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    window_size: int,
    overlap: int = 0,
    drop_partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Calculate the corner points of all windows covering the area from min_x to max_x and min_y to max_y,
    with specified window_size and overlap.

    Parameters
    ----------
    min_x: float
        minimum X coordinate
    max_x: float
        maximum X coordinate
    min_y: float
        minimum Y coordinate
    max_y: float
        maximum Y coordinate
    window_size: float
        size of each window
    overlap: float
        overlap between consecutive windows (must be less than window_size)
    drop_partial_windows: bool
        if True, drop border windows that are smaller than window_size;
        if False, create smaller windows at the borders to cover the remaining space.

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    x_step = window_size - overlap
    y_step = window_size - overlap

    # Generate starting points
    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    # Adjust windows that extend beyond the bounds
    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]

# a helper function for "_sliding_window_psudobulk"
# hugely based on the sliding_function in squidpy
# https://github.com/scverse/squidpy/blob/afcb8d0e81085a1af03ae5a9f299c5df00e95d61/src/squidpy/tl/_sliding_window.py
# modified to output coordinates of each window
def _sliding_window(
    adata: AnnData | SpatialData,
    library_key: str | None = None,
    window_size: int | None = None,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool = False,
    copy: bool = False,
):
    """
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    window_size: int
        Size of the sliding window.
    %(library_key)s
    coord_columns: Tuple[str, str]
        Tuple of column names in `adata.obs` that specify the coordinates (x, y), e.i. ('globalX', 'globalY')
    sliding_window_key: str
        Base name for sliding window columns.
    overlap: int
        Overlap size between consecutive windows. (0 = no overlap)
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
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    if isinstance(adata, SpatialData):
        adata = adata.table

    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    # extract coordinates of observations
    x_col, y_col = coord_columns
    if x_col in adata.obs and y_col in adata.obs:
        coords = adata.obs[[x_col, y_col]].copy()
    elif spatial_key in adata.obsm:
        coords = pd.DataFrame(
            adata.obsm[spatial_key][:, :2],
            index=adata.obs.index,
            columns=[x_col, y_col],
        )
    else:
        raise ValueError(
            f"Coordinates not found. Provide `{coord_columns}` in `adata.obs` or specify a suitable `spatial_key` in `adata.obsm`."
        )

    # infer window size if not provided
    if window_size is None:
        coord_range = max(
            coords[x_col].max() - coords[x_col].min(),
            coords[y_col].max() - coords[y_col].min(),
        )
        # mostly arbitrary choice, except that full integers usually generate windows with 1-2 cells at the borders
        window_size = max(int(np.floor(coord_range // 3.95)), 1)

    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")

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
            overlap=overlap,
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

            if overlap == 0:
                mask = (
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] <= x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] <= y_end)
                )
                obs_indices = lib_coords.index[mask]
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"
                
                ############ added to the original function in squidpy ############
                # to get coordinates of each window
                # center point
                sliding_window_df.loc[obs_indices, 'window_col'] = x_start+(window_size/2)
                sliding_window_df.loc[obs_indices, 'window_row'] = y_start+(window_size/2)
                ###################################################################

            else:
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                sliding_window_df.loc[obs_indices, col_name] = True
                sliding_window_df.loc[:, col_name].fillna(False, inplace=True)

    if overlap == 0:
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
    
    ############ added to the original function in squidpy ############
    # report total window number and average cell number per window
    n_windows = len(set(sliding_window_df[sliding_window_key]))
    print(f'### Total window number ###: {n_windows}')
    print(f'### Average cell number per window ###: {round(sliding_window_df.shape[0]/n_windows,1)}')
    ###################################################################

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)

# this is a helper function for "preprocess"
def _sliding_window_psudobulk(
    adata: AnnData,
    section_col,
    window_size: int,
    coord_columns: Optional[Tuple[str, str]], # if None, need to have XY coordinate in the adata.obsm['spatial']
    log_normalise=True
):
    
    ### create slinding_windows
    # this will add columns related to the assigned windows in the adata.obs
    if coord_columns!=None:
        _sliding_window(
            adata=adata,
            library_key=section_col,  # only one section in the object
            window_size=window_size,
            overlap=0,
            coord_columns=coord_columns,
            copy=False,  # we modify in place
        )
    else:
        _sliding_window(
            adata=adata,
            library_key=section_col,  # only one section in the object
            window_size=window_size,
            overlap=0,
            coord_columns=("globalX", "globalY"),
            spatial_key='spatial',
            copy=False,  # we modify in place
        )
    
    ### pseudobulk per window
    # aggregate counts per window
    bdata = sc.get.aggregate(adata, by='sliding_window_assignment', func=["sum"])
    bdata.X = bdata.layers['sum'].copy()

    # transfer important obs data
    obs_sel = adata.obs[[section_col,'window_col','window_row','sliding_window_assignment']].copy()
    obs_sel.drop_duplicates(inplace=True)
    obs_sel.set_index('sliding_window_assignment',inplace=True)
    bdata.obs[[section_col,'window_col','window_row']] = obs_sel.loc[bdata.obs_names].values

    # log-normalise
    if log_normalise:
        sc.pp.normalize_total(bdata, target_sum=1e4)
        sc.pp.log1p(bdata)
        print('Pseudobulk by summing per window and then log-normalising.')
    else:
        print('Pseudobulk by summing per window. The output is not normalised.')
              
    return bdata

# a helper function for "prepare_dataframe"
def _include_neighbours(
    data: pd.DataFrame,
    k: int = 6
) -> pd.DataFrame:
    """
    Augment each spot's gene expression features by including neighbor-based summary statistics.
    For each section in the DataFrame, computes the maximum expression among k nearest neighbors
    for each gene, and returns an augmented DataFrame with original and neighbor features.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by spot ID, containing 'x', 'y' coordinates, gene expression columns
        ending with '_own', and a 'section' column identifying sections.
    k : int, default=6
        Number of neighbors to consider (excluding self).

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with original and neighbor-max features, plus 'section', 'x', 'y',
        and optionally 'tissue' columns.
    """
    
    # Identify expression feature columns
    expression_features = [col for col in data.columns if col.endswith('_own')]
    
    # Process each section separately
    augmented_df_list=[]
    for section, group_df in data.groupby('section'):
        # Extract spatial coordinates and gene expression for this section
        datapoint_ids = group_df.index
        coords = group_df[['x', 'y']].values
        X_gene = group_df[expression_features].values  # shape: (n_datapoints, n_genes)
        n_datapoints, n_genes = X_gene.shape

        # Set number of neighbors (excluding self); adjust if the section has too few spots
        k_section = k if n_datapoints > k + 1 else max(n_datapoints - 1, 1)

        # Find neighbors including self, then exclude self later
        nbrs = NearestNeighbors(n_neighbors=k_section + 1, algorithm='ball_tree')
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)

        # Compute max expression among neighbors for each spot
        neighbor_max = np.zeros_like(X_gene)
        for i in range(n_datapoints):
            neighbor_idx = indices[i][1:]  # Exclude the spot itself
            neighbor_values = X_gene[neighbor_idx, :]  # shape: (k_section, n_genes)
            neighbor_max[i, :] = neighbor_values.max(axis=0)
        
        # Prepare column names for neighbor features
        max_cols = [col.replace('_own', '_neighbour-max') for col in expression_features]

        # Combine original and neighbor-max features
        X_aug = np.hstack([X_gene, neighbor_max])      
        all_cols = expression_features + max_cols
        section_df = pd.DataFrame(X_aug, columns=all_cols, index=datapoint_ids)

        # Re-add metadata columns
        section_df['section'] = section
        section_df[['x', 'y']] = group_df[['x', 'y']]
        if 'tissue' in group_df.columns:
            section_df['tissue'] = group_df['tissue'].values
        
        # append
        augmented_df_list.append(section_df)
    
    # Concatenate all sections
    augmented_data = pd.concat(augmented_df_list)
    return augmented_data

# this is a helper function for "_find_technical_edge"
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

# this is a helper function for "_annotate_edge"
def _find_technical_edge(data):
    # get ids which are at the four side
    x_max_ids = data.index[data['x']==data['x'].max()]
    x_min_ids = data.index[data['x']==data['x'].min()]
    y_max_ids = data.index[data['y']==data['y'].max()]
    y_min_ids = data.index[data['y']==data['y'].min()]
    # get ids with arithmetic progression, at the four side 
    xmax_ap_ids_list = _find_arithmetic_segments(data.loc[x_max_ids,'y'].sort_values())
    xmin_ap_ids_list = _find_arithmetic_segments(data.loc[x_min_ids,'y'].sort_values())
    ymax_ap_ids_list = _find_arithmetic_segments(data.loc[y_max_ids,'x'].sort_values())
    ymin_ap_ids_list = _find_arithmetic_segments(data.loc[y_min_ids,'x'].sort_values())
    # return
    technical_edge_ids = xmax_ap_ids_list+xmin_ap_ids_list+ymax_ap_ids_list+ymin_ap_ids_list
    return technical_edge_ids

# this is a helper function for "_annotate_edge"
def _distance_to_edge(data,
                     norm=True,
                     log1p=True,
                    ):
    # Separate the edge spots from the others
    edge_df = data[data['is_edge'] == True].copy()
    non_edge_df = data[data['is_edge'] == False].copy()

    # Build the kNN reference using the edge spots
    # Using n_neighbors=1 to get the single nearest edge spot for each query point
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(edge_df[['x', 'y']])

    # For each non-edge spot, find the nearest edge spot
    distances, indices = nbrs.kneighbors(non_edge_df[['x', 'y']])
    non_edge_df['distance_to_edge'] = distances.flatten()

    # put them in the original dataframe
    data.loc[non_edge_df.index,'distance_to_edge'] = non_edge_df['distance_to_edge']
    data.loc[edge_df.index,'distance_to_edge'] = 0
    
    # min-max normalisation
    if norm:
        scaler = MinMaxScaler()
        data['distance_to_edge'] = scaler.fit_transform(data['distance_to_edge'].to_numpy().reshape(-1, 1))

    # transform distance
    # scale --> log(1 + x)
    if log1p:
        data['distance_to_edge'] = data['distance_to_edge']*10 # this scale factor determines how much emphasis you want to have for a small difference at the epicardial side
        data['distance_to_edge'] = np.log1p(data['distance_to_edge'])

    return data

# this is a helper function for "preprocess"
def _annotate_edge(data, tile_type, remove_technical_edge, plot):
    ### check 'title_type' argument
    if tile_type not in ('hexagon','square'):
        raise ValueError("Invalid title_type value. Expected 'hexagon' or 'square'.")
    ### Initialize new columns
    data['n_neighbours'] = np.nan
    data['is_edge'] = False
    data['distance_to_edge'] = np.nan
    ### annotate "edge" data per section
    for section, df in data.groupby('section'):
        coords = df[['x','y']].values
        if tile_type=='hexagon':
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
        if tile_type=='hexagon':
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
        df = _distance_to_edge(df,norm=True,log1p=True)
        data.loc[df.index,'distance_to_edge'] = df['distance_to_edge']
        # plot for checking
        if plot:
            sns.scatterplot(
                x='x',y='y',hue='distance_to_edge',data=df,
                palette='rainbow_r',s=10
           )
            plt.legend(title='distance to edge\n(normalised)',bbox_to_anchor=(1, 1))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(section)
            plt.show()
    return data


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
        bdata = _sliding_window_psudobulk(
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
    data = _annotate_edge(data, # dataframe obtained by running "prepare_dataframe"
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


