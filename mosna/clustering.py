#! /usr/bin/env python3
"""mosna.clustering — dimensionality reduction, clustering and cluster post-processing

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403

from mosna.utils import to_numpy  # noqa: F401


def make_cluster_cmap(labels, grey_pos='end', saturated_first=True, as_mpl_cmap=False):
    """
    Creates an appropriate colormap for a vector of cluster labels.
    
    Parameters
    ----------
    labels : array_like
        The labels of multiple clustered points
    grey_pos: str
        Where to put the grey color for the noise
    
    Returns
    -------
    cmap : matplotlib colormap object
        A correct colormap
    
    Examples
    --------
    >>> my_cmap = make_cluster_cmap(labels=np.array([-1,3,5,2,4,1,3,-1,4,2,5]))
    """    

    cmap = ['#1F77B4', '#FF7F0E',  '#2CA02C', '#D62728', '#9467BD',
            '#8C564B', '#17BECF', '#E377C2', '#BCBD22', '#7F7F7F']
    if grey_pos == 'start':
        cmap[0], cmap[-1] = cmap[-1], cmap[0]
    if len(labels) > 10:
        cmap_next = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
                      '#C49C94', '#9EDAE5', '#F7B6D2', '#DBDB8D', '#C7C7C7']
        if grey_pos == 'start':
            cmap_next[0], cmap_next[-1] = cmap_next[-1], cmap_next[0]
        # select as few as lowly saturated colors as possible
        if len(labels) < 20:
            cmap_next = cmap_next[:len(labels)-10]
        if saturated_first:
            cmap = cmap + cmap_next
        else:
            # put the saturated colors at the end
            cmap = cmap_next + cmap
    if len(labels) > 20:
        cmap_end = ['#00FFFF', '#00FF00', '#FF00FF', '#FF007F']
        cmap = cmap + cmap_end
    if as_mpl_cmap:
        # TODO: check if hex convert to tuple of size 3
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap)
    
    return cmap


def aggregate_cell_types(
    var_aggreg_samples_info: pd.DataFrame,
    cohort_data: pd.DataFrame,
    pheno_col: str,
    patient_col: str,
    sample_col: str,
    nodes_dir: Path = None,
    file_name: str = 'cell_types.npy',
    save_data: bool = True,
    force_recompute: bool = False,
    ):
    """
    Aggregate cell types in the same order of patients and samples
    IDs as for the Neighbors Aggregation Statistics method.

    Parameters
    ----------
    var_aggreg : pd.DataFrame
        Aggregated statistics of omics data for each cell's neighborhood.
    cohort_data : pd.DataFrame
        Data from the cohort per cell, including patients and samples IDs, 
        and cell types.
    pheno_col : str
        Column name of cell types.
    patient_col : str
        Column name of patients IDs
    sample_col : str
        Column name of samples IDs
    nodes_dir : Path or None
        Path to nodes data directory.
    file_name : str
        Name for the aggregated cell types file.
    save_data : bool
        If True, save aggregated data to disk.
    force_recompute : bool
        If True, recompute aggregated cell types even if 
        present on disk.

    Returns
    -------
    cell_types : np.array
        Numpy array of cell types.        
    """

    if nodes_dir is not None:
        path_cell_types = nodes_dir / file_name
        if path_cell_types.exists() and not force_recompute:
            print("Loading cell types in correct order")
            cell_types = np.load(path_cell_types, allow_pickle=True)
            return cell_types

    print("Aggregating cell types in correct order")
    # pairs of patient id and sample id
    uniq_pairs = var_aggreg_samples_info.drop_duplicates()

    all_cell_types = []
    for idx, patient_id, sample_id in tqdm(uniq_pairs.itertuples()):
        cell_types = cohort_data.loc[
                        (cohort_data[patient_col] == patient_id) &
                        (cohort_data[sample_col] == int(sample_id)),
                        pheno_col,
                        ]
        all_cell_types.append(cell_types.values)
    cell_types = np.hstack([*all_cell_types])
    print(f'Concatenated {cell_types.size} cells')
    
    if save_data:
        if nodes_dir is not None:
            path_cell_types = nodes_dir / file_name
            np.save(path_cell_types, cell_types)
        else:
            print("Provide `nodes_dir` to save aggregated cell types data.")
    
    return cell_types


def make_niches_composition(var, niches, var_label='variable', normalize='total'):
    """
    Make a counts matrix of cell types composition of niches.
    """
    df = pd.DataFrame({var_label: var,
                       'niches': niches})
    df['counts'] = np.arange(df.shape[0])
    counts = df.groupby([var_label, 'niches']).count()
    counts = counts.reset_index().pivot(
        index=var_label, 
        columns='niches', 
        values='counts').fillna(0)
    if normalize == 'total':
        counts = counts / df.shape[0]
    elif normalize == 'obs':
        # pandas has some unconvenient bradcasting behaviour otherwise
        counts = counts.div(counts.sum(axis=1), axis=0)
    elif normalize == 'niche':
        counts = counts / counts.sum(axis=0)
    elif normalize == 'clr':
        X = counts.values
        # avoid null values
        X[X == 0] = X.max() / 100000
        # CLR tranformation
        X_clr = cs.clr(cs.closure(X))
        counts.loc[:, :] = X_clr
    elif normalize == 'niche&obs':
        counts = counts.div(counts.sum(axis=1), axis=0)
        counts = counts / counts.sum(axis=0)
    
    return counts


def make_reducer_name(
    reducer_type,
    n_dimensions=None,
    n_neighbors=None,
    metric=None,
    min_dist=None,
    ):
    if reducer_type == 'umap':
        reducer_name = f"reducer-{reducer_type}_dim-{n_dimensions}_nneigh-{n_neighbors}_metric-{metric}_min_dist-{min_dist}"
    elif reducer_type == 'none':
        reducer_name = f"reducer-{reducer_type}"
    return reducer_name


def get_reducer(
    data, 
    data_dir, 
    reducer_type='umap', 
    n_components=2, 
    n_neighbors=15, 
    metric='manhattan', 
    min_dist=0.0, 
    force_recompute=False,
    save_reduced_coords=True, 
    save_reducer=False,
    return_path_coords=False, 
    random_state=None, 
    verbose=1,
    ):
    """
    Generate or load a dimensionality reduction (DR) model and transformed (reduced) data.

    Parameters
    ----------
    data : ndarray
        Dataset on which we want to apply the DR method.
    data_dir : str or pathlib Path object
        Directory where the DR model and transformed data are stored.
    reducer_type : str
        DR method, can be 'umap', for now, other ones coming soon.
    n_components : int
        Number of final dimensions.
    n-neighbors : int
        Number of closest neighbors used in various DR methods.
    metric : str
        Type of distance used, like 'manhattan', 'euclidean' or 'cosine'.
    min_dist : float
        Minimum distance between DRed data, we usually want 0.
    save_reducer : bool
        Whether the reducer object is saved. 
    random_state : int
        Controls the random initialization of the DR model.
    
    Returns
    -------
    embedding : ndarray
        Reduced coordinates of the dataset.
    reducer : object
        The DR model, its type depends on the choosen DR method.

    Example
    -------
    In the *mosna* pipeline, `var_aggreg` is the array of aggregated statistics:
    >>> embedding, reducer = get_reducer(data=var_aggreg, data_dir=nas_dir)
    """

    reducer_name = make_reducer_name(reducer_type, n_components, n_neighbors, metric, min_dist)
    data_dir = Path(data_dir) / reducer_name
    file_path = data_dir / "embedding"
    if os.path.exists(str(file_path) + '.npy') and not force_recompute:
        if verbose > 0: 
            print("Loading reducer object and reduced coordinates")
        embedding = np.load(str(file_path) + '.npy')
        if os.path.exists(str(data_dir / "reducer") + '.pkl'):
            reducer = joblib.load(str(data_dir / "reducer") + '.pkl')
        else:
            reducer = None
    else:
        if verbose > 0: 
            print("Computing dimensionality reduction")
        if reducer_type == 'umap':
            n_neighbors = int(n_neighbors)
            reducer = UMAP(
                random_state=random_state,
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric=metric,
                min_dist=min_dist,
                )
            if isinstance(data, pd.DataFrame):
                embedding = reducer.fit_transform(data.values)
            else:
                embedding = reducer.fit_transform(data)
        elif reducer_type == 'none':
            reducer = {'reducer_type': 'none'}
            if isinstance(data, pd.DataFrame):
                embedding = data.values
            else:
                embedding = data

        path_coords = str(file_path) + '.npy'
        if save_reduced_coords:
            # save reduced coordinates
            data_dir.mkdir(parents=True, exist_ok=True)
            np.save(path_coords, embedding, allow_pickle=False, fix_imports=False)
        if save_reducer:
            # save the reducer object
            joblib.dump(reducer, str(data_dir / "reducer") + '.pkl')
    
    if return_path_coords:
        return embedding, reducer, path_coords
    return embedding, reducer


def get_clusterer(
        data,
        data_dir,
        reducer_type='umap', 
        n_neighbors=15, 
        metric='manhattan', 
        min_dist=0.0,
        clusterer_type='leiden', 
        dim_clust=2, 
        k_cluster=15, 
        resolution=0.005,
        resolution_parameter=None,
        n_clusters=None,
        ecg_min_weight=0.05, 
        ecg_ensemble_size=20,
        min_cluster_size=0.001,
        noise_to_cluster=False,
        flavor=None,
        avoid_neigh_overflow=True,
        force_recompute=False, 
        use_gpu=True,
        random_state=None,
        save_net_data=True,
        verbose=1,
        ):
    """
    Generate or load a clustering model and cluster labels.

    Parameters
    ----------
    data : ndarray
        Dataset on which we want to apply the DR method.
    data_dir : str or pathlib Path object
        Directory where the DR model and transformed data are stored.
    reducer_type : str
        DR method, can be 'umap', for now, other ones coming soon.
    n_components : int
        Number of final dimensions.
    n-neighbors : int
        Number of closest neighbors used in various DR methods.
    metric : str
        Type of distance used.
    min_dist : float
        Minimum distance between DRed data, we usually want 0.
    clusterer_type : str
        Clustering algorithm to partition data, either 'leiden', 'ecg' for Ensemble 
        Clustering for Graphs, 'spectral' for balanced spectral clustering or 'gmm' 
        for Gaussian Mixture Model.
    dim_clust : int
        Dimensionality of the reducede space in which data is clustered.
        A higher number allows for more complex cluster shapes, but introduces outliers.
    k_cluster : int
        Number of neighbors considered during the clustering.
    resolution : float
        Level of details of the clustering. A higher number increases the level of details.
    n_clusters : int, None
        Number of target clusters, used with GaussianMixtureModel clusterer.
    ecg_min_weight : float, 0.05
        min_weight parameter for the ecg method.
    ecg_ensemble_size : int, 20
        ensemble_size parameter for the ecg method.
    flavor : str, None
        If 'CellCharter', uses UMAP for dimensionality reduction, and a gaussian mixture
        model for clustering. 
    avoid_neigh_overflow : bool, True
        Whether the number of neighbors for clustering is limited by the number of
        neighbors for dimensionality reduction.
    force_recompute : bool
        Whether computation occurs even if results already exist in `data_dir`.
    use_gpu : boo
        If True, GMM clustering leverages GPU.
    
    Returns
    -------
    embedding : ndarray
        Reduced coordinates of the dataset.
    reducer : object
        The DR model, its type depends on the choosen DR method.

    Example
    -------
    >>> np.random.seed(0)
    >>> data = np.random.rand(800, 4)
    >>> cluster_labels, cluster_dir, nb_clust, G = get_clusterer(data, "test")
    """
    n_neighbors = int(n_neighbors)
    dim_clust = int(dim_clust)
    min_dist = float(min_dist)          
    
    # API compatibility
    if resolution_parameter is not None:
        resolution_parameter = float(resolution_parameter)  
        resolution = resolution_parameter

    if flavor is not None:
        if flavor == 'UTAG':
            print('not implemented yet')
        elif flavor == 'CellCharter':
            reducer_type = 'umap'
            clusterer_type = 'gmm'
            if n_clusters is None:
                n_clusters = 10
    reducer_name = make_reducer_name(reducer_type, dim_clust, n_neighbors, metric, min_dist)
    reducer_dir = Path(data_dir) / reducer_name
    k_cluster = int(k_cluster)
    if avoid_neigh_overflow and k_cluster > n_neighbors:
        if verbose > 0:
            print('setting k_cluster = {k_cluster} to n_neighbors: {n_neighbors}')
        k_cluster = n_neighbors

    if clusterer_type == "leiden":
        cluster_dir = reducer_dir / f"clusterer-{clusterer_type}_n_neighbors-{k_cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        clusterer_name = f"leiden_resolution-{resolution}"
        # knn network in reduced space, common to several clustering methods:
        reduced_net_path = reducer_dir / f'edges_n_neighbors-{k_cluster}.parquet'
    elif clusterer_type == "ecg":
        cluster_dir = reducer_dir / f"clusterer-{clusterer_type}_n_neighbors-{k_cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        clusterer_name = f"ecg_min_weight-{ecg_min_weight}_ensemble_size-{ecg_ensemble_size}"
        reduced_net_path = reducer_dir / f'edges_n_neighbors-{k_cluster}.parquet'
    elif clusterer_type == "spectral":
        assert n_clusters is not None
        cluster_dir = reducer_dir / f"clusterer-{clusterer_type}_n_neighbors-{k_cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        clusterer_name = f"spectral_n_clusters-{n_clusters}"
        reduced_net_path = reducer_dir / f'edges_n_neighbors-{k_cluster}.parquet'
    elif clusterer_type == "hdbscan":
        assert min_cluster_size is not None
        cluster_dir = reducer_dir / f"clusterer-{clusterer_type}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        clusterer_name = f"hdbscan_min_cluster_size-{min_cluster_size}_noise_to_cluster-{noise_to_cluster}"
    elif clusterer_type == "gmm":
        assert n_clusters is not None
        cluster_dir = reducer_dir / f"clusterer-{clusterer_type}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        clusterer_name = f"gmm_n_clusters-{n_clusters}"
    file_path = cluster_dir / clusterer_name

    if os.path.exists(str(file_path) + '_labels.npy') and not force_recompute:
        # load clustered data
        if verbose > 0: 
            print("Loading clusterer object and cluster labels")
        cluster_labels = np.load(str(file_path) + '_labels.npy')
        nb_clust = len(np.unique(cluster_labels))
        if verbose > 0: 
            print(f"There are {nb_clust} clusters")
        
        # load estimator or network data for return
        if clusterer_type in ['leiden', 'ecg', 'spectral']:
            try:
                if gpu_clustering:
                    edges = cudf.read_parquet(reduced_net_path)
                else:
                    edges = pd.read_parquet(reduced_net_path)
            except FileNotFoundError:
                edges = None
        elif clusterer_type in ["hdbscan", "gmm"]:
            try:
                clusterer = joblib.load(str(file_path) + '_clusterer.joblib')
            except FileNotFoundError:
                clusterer = None
        save_net_data = False
    else:
        # get the embedding of data
        embedding, _ = get_reducer(
            data, data_dir, reducer_type, dim_clust, n_neighbors, 
            metric, min_dist, random_state=random_state, verbose=verbose)
        if verbose > 0: 
            print("Performing clustering")

        if clusterer_type in ['leiden', 'ecg', 'spectral']:
            if reduced_net_path.exists():
                if verbose > 1:
                    print('loading knn graph')
                if gpu_clustering:
                    edges = cudf.read_parquet(reduced_net_path)
                    # send edges to GPU, with dummy weights
                    G = cugraph.Graph()
                    G.from_cudf_edgelist(
                        edges, 
                        source='src', 
                        destination='dst', 
                        edge_attr='weight', 
                        renumber=False,
                        )
                else:
                    edges = pd.read_parquet(reduced_net_path)
                    embedding_pairs = edges[['src', 'dst']].values
                    G = ty.to_iGraph(embedding, embedding_pairs)
            else:
                # need to build knn graph
                if verbose > 1:
                    print('building knn graph')
                # from the UMAP documentation:
                # "By default UMAP embeds data into Euclidean space"
                # so the clusterer should use the Euclidean metric
                embedding_pairs = ty.build_knn(embedding, k=k_cluster, metric='euclidean')

                if gpu_clustering:
                    # send edges to GPU, with dummy weights
                    edges_np = np.hstack((embedding_pairs, np.ones((len(embedding_pairs), 1)))).astype(np.int32)
                    edges = cudf.DataFrame(edges_np, columns=['src', 'dst', 'weight'])
                    G = cugraph.Graph()
                    G.from_cudf_edgelist(
                        edges, 
                        source='src', 
                        destination='dst', 
                        edge_attr='weight', 
                        renumber=False,
                        )
                else:
                    edges_np = np.hstack((embedding_pairs, np.ones((len(embedding_pairs), 1)))).astype(np.int32)
                    edges = pd.DataFrame(edges_np, columns=['src', 'dst', 'weight'])
                    G = ty.to_iGraph(embedding, embedding_pairs)
                    
            if clusterer_type == "leiden":
                if gpu_clustering:
                    if verbose > 1:
                        print("performing leiden clustering on GPU")    
                    partition, modularity_score = cugraph.leiden(G, max_iter=100, resolution=resolution)
                    cluster_labels = partition['partition'].values
                else:
                    if verbose > 1:
                        print("performing leiden clustering on CPU")
                    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=0)
                    # or other partition such as la.RBERVertexPartition
                    cluster_labels = np.array(partition.membership)

            elif clusterer_type == "ecg":
                if gpu_clustering:
                    if verbose > 1:
                        print("performing ECG clustering on GPU")
                    partition = cugraph.ecg(G, min_weight=ecg_min_weight, ensemble_size=ecg_ensemble_size)
                    cluster_labels = partition['partition'].values
                else:
                    raise RuntimeError('ecg clustering requires the cugraph library')

            elif clusterer_type == "spectral":
                if gpu_clustering:
                    if verbose > 1:
                        print("performing spectral clustering on GPU")
                    partition = cugraph.spectralBalancedCutClustering(G, n_clusters)
                    cluster_labels = partition['cluster'].values
                else:
                    if verbose > 1:
                        print("performing spectral clustering on CPU")
                    from sklearn.cluster import SpectralClustering
                    cluster_labels = SpectralClustering(
                        n_clusters=n_clusters, 
                        assign_labels='discretize', 
                        random_state=0,
                        ).fit_predict(embedding)

        elif clusterer_type == 'hdbscan':
            if min_cluster_size < 1:
                min_cluster_size = int(min_cluster_size * len(embedding))
            args_clust = {}
            if not gpu_clustering:
                args_clust['core_dist_n_jobs'] = cpu_count()
            
            if noise_to_cluster:
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size, 
                    min_samples=None, 
                    prediction_data=True, 
                    **args_clust,
                )
                clusterer.fit(embedding)
                soft_clusters = all_points_membership_vectors(clusterer)
                if len(soft_clusters.shape) > 1:
                    cluster_labels = soft_clusters.argmax(axis=1)
                else:
                    cluster_labels = soft_clusters
            else:
                clusterer = HDBSCAN( 
                    min_cluster_size=min_cluster_size, 
                    min_samples=1,
                    **args_clust,
                )
                clusterer.fit(embedding)
                cluster_labels = clusterer.labels_

        elif clusterer_type == "gmm":
            if use_gpu:
                if verbose > 1:
                    print("performing GMM clustering on GPU")
                clusterer = GaussianMixture(n_clusters, trainer_params=dict(accelerator='gpu', devices=1))
            else:
                if verbose > 1:
                    print("performing GMM clustering on CPU")
                clusterer = GaussianMixture(n_clusters)
            # make cluster predictions
            clusterer.fit(embedding.astype(np.float32))
            cluster_labels = np.array(clusterer.predict(embedding.astype(np.float32)))
            
        # make sure cluster_labels is numpy array
        cluster_labels = to_numpy(cluster_labels)

        nb_clust = len(np.unique(cluster_labels))
        if verbose > 0: 
            print(f"Found {nb_clust} clusters")
        # save cluster labels
        np.save(str(file_path) + '_labels.npy', cluster_labels, allow_pickle=False)
    if clusterer_type in ["leiden", "ecg", "spectral"]:
        if save_net_data:
            edges.to_parquet(reduced_net_path)
        return cluster_labels, cluster_dir, nb_clust, edges
    elif clusterer_type in ["hdbscan", "gmm"]:
        if save_net_data:
            joblib.dump(clusterer, str(file_path) + '_estimator.joblib')
        return cluster_labels, cluster_dir, nb_clust, clusterer


def relabel_clusters(
    clusters: np.array,
) -> np.array:
    """
    Relabel the N clusters to have ids between 0 and N-1.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    
    Returns
    -------
    new_clusters : np.array
        Potentially relabelled clusters.
    """
    bins, counts = np.unique(clusters, return_counts=True)
    if len(bins) == bins.max() - 1:
        return clusters
    else:
        new_bins = np.arange(len(bins))
        new_clusters = np.zeros_like(clusters)
        for i in range(len(new_bins)):
            new_clusters[clusters == bins[i]] = new_bins[i]
        return new_clusters


def merge_clusters(
    clusters: np.array,
    coords: np.ndarray,
    size_thresh: Union[int, None] = None,
    size_perc: int = 25,
    ratio_size: float = 0.1,
    n_neigh_max: int = 10,
    force_merge: bool = False,
    verbose: int = 1,
) -> Tuple[np.array, bool]:
    """
    Merge the smallest cluster to it's closest cluster if its size
    is lower than a given size.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    coords : np.ndarray
        Coordinates of points in clusters.
    size_thresh : Union[int, None], None
        If provided, the size threshold to merge the smallest cluster.
        If none, computed as the base size * ratio_size
    size_perc : int, 25
        Percentile of cluster size as the base size used for the size threshold.
    ratio_size : float, 0.1
        Ratio to compute size_thresh.
    n_neigh_max : int, 10
        Number of points to consider inside the cluster to merge for the closest neighbor.
    force_merge : bool, False
        If True, the smallest cluster is merged even if it is big enough.
    verbose : int, 1
        Verbosity level.
    
    Returns
    -------
    clusters : np.array
        Potentially merged clusters.
    merged : bool
        Whether a merge occured.
    """

    merged = False
    cluster_ids, cluster_sizes = np.unique(clusters, return_counts=True)
    if len(cluster_ids) == 1:
        return clusters, merged
    smallest_id = np.argmin(cluster_sizes)
    if size_thresh is None:
        size_thresh = np.percentile(cluster_sizes, size_perc) * ratio_size
    if force_merge or cluster_sizes[smallest_id] < size_thresh:
        select = clusters == cluster_ids[smallest_id]
        coords_in = coords[select, :]
        coords_out = coords[~select, :]
        clusters_out = clusters[~select]
        n_neigh_max = min(n_neigh_max, cluster_sizes[smallest_id])

        kdt = cKDTree(coords_out)
        # closest point id
        dist, pairs = kdt.query(x=coords_in, k=1)

        # Get the closest points to another cluster (avoid inner points)
        closest_neigh_ids = np.argsort(dist)[:n_neigh_max]
        closest_clusters = clusters_out[pairs][closest_neigh_ids]
        val, counts = np.unique(closest_clusters, return_counts=True)
        closest_cluster = val[np.argmax(counts)]
        # make a copy of clusters
        clusters = np.array(clusters)
        clusters[select] = closest_cluster
        if verbose:
            print(f'cluster {cluster_ids[smallest_id]} merged with cluster {closest_cluster}')
        merged = True
    return clusters, merged


def merge_clusters_until(
    clusters: np.array,
    coords: np.ndarray,
    cond_n_clust: Union[int, None] = None,
    force_n_clust: bool = False,
    size_thresh: Union[int, None] = None,
    size_perc: int = 25,
    ratio_size: float = 0.1,
    n_neigh_max: int = 10,
    relabel_clusters_ids: bool = True,
    verbose: int = 1,
) -> np.array:
    """
    Merge iteratively the smallest cluster to it's closest cluster 
    if its size is lower than a given size, until no further merging
    occurs or until a condition is reached.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    coords : np.ndarray
        Coordinates of points in clusters.
    cond_n_clust: Union[int, None], None
        Sufficient condition on the number of clusters below which the iterative
        merge stops.
    force_n_clust : bool, False
        If True, force merging until the desired number of clusters is reached.
        It overides the 'until no further merging occurs' condition.
    size_thresh : Union[int, None], None
        If provided, the size threshold to merge the smallest cluster.
        If none, computed as the base size * ratio_size
    size_perc : int, 25
        Percentile of cluster size as the base size used for the size threshold.
    ratio_size : float, 0.1
        Ratio to compute size_thresh.
    n_neigh_max : int, 10
        Number of points to consider inside the cluster to merge for the closest neighbor.
    relabel_clusters_ids : bool, True
        If True, relabel N clusters between 0 and N-1.
    verbose : int, 1
        Verbosity level.
    
    Returns
    -------
    clusters : np.array
        Potentially merged clusters.
    """

    keep_merging = True
    while keep_merging:
        clusters, merged = merge_clusters(
            clusters=clusters,
            coords=coords,
            size_thresh=size_thresh,
            size_perc=size_perc,
            ratio_size=ratio_size,
            n_neigh_max=n_neigh_max,
            force_merge=force_n_clust,
            verbose=verbose,
        )

        if not merged and not force_n_clust:
            keep_merging = False
            if verbose > 0:
                print('no further merging can occur')
        else:
            bins = np.unique(clusters)
            if cond_n_clust is not None and len(bins) <= cond_n_clust:
                keep_merging = False
                if verbose > 0:
                    print(f'maximum number of clusters {cond_n_clust} reached')
    if relabel_clusters_ids:
        clusters = relabel_clusters(clusters)
    return clusters
