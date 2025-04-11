import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

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

# this is a helper function for "annotate_edge"
def _find_technical_edge(data): # dataframe obtained by running "prepare_dataframe"
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

# this is a helper function for "annotate_edge"
def _distance_to_edge(data, # dataframe obtained by running "annotate_edge"
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

def annotate_edge(data, tile_type, remove_technical_edge, plot):
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

