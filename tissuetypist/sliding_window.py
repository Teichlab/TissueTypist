# hugely based on the sliding_function in squidpy
# https://github.com/scverse/squidpy/blob/afcb8d0e81085a1af03ae5a9f299c5df00e95d61/src/squidpy/tl/_sliding_window.py
# modified to output coordinates of each window

from itertools import product
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from scanpy import logging as logg
from spatialdata import SpatialData
from squidpy.gr._utils import _save_data
from typing import Optional, Tuple

# a helper function for "_sliding_window"
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

# a helper function for "sliding_window_psudobulk"
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

def sliding_window_psudobulk(
    adata: AnnData,
    section_col,
    window_size: int,
    coord_columns: Optional[Tuple[str, str]], # if None, need to have XY coordinate in the adata.obsm['spatial']
    log_normalise=True
):
    
    ### create slinding_windows
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
    
        