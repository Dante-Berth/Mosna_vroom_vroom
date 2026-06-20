#! /usr/bin/env python3
"""mosna.features — spatial-omic feature / niche computation

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403
from mosna import io  # fast Polars/Parquet I/O

from mosna.neighbors import make_features_NAS  # noqa: F401
from mosna.preprocessing import make_data_index  # noqa: F401


def make_features_STAGATE(
    X: np.array, 
    pairs: np.array, 
    var_names: Union[Iterable[str], None] = None,
    ) -> pd.DataFrame:
    """
    Compute feature vectors of each node in a network
    given the STAGATE method.

    Parameters
    ----------
    X : array_like
        Nodes' attributes on which features are computed.
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    var_names : list
        Names of variables of X.

    Returns
    -------
    feats : dataframe
        Features computed with the STAGATE method.
    """
    # code here
    pass


def make_features_SCANIT(
    X: np.array = None, 
    coords: np.array = None, 
    pairs: np.array = None, 
    adata: ad.AnnData = None,
    var_names: Union[Iterable[str], None] = None,
    spatial_graph_kwargs: Union[dict, None] = None,
    spatial_representation: Union[dict, None] = None,
    return_anndata: bool = False,
    ) -> pd.DataFrame:
    """
    Compute feature vectors of each node in a network
    given the SCAN-IT method.

    Parameters
    ----------
    X : array_like
        Nodes' attributes on which features are computed.
    coords : array_like
        Coordinates of cells.
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    var_names : list
        Names of variables of X.

    Returns
    -------
    feats : dataframe
        Features computed with the SCAN-IT method.
    """
    import scanit

    if adata is None:
        adata = ty.add_to_AnnData(
            coords, 
            pairs, 
            adata=None,
            counts=X,
            obs_names=None,
            var_names=var_names,
            return_adata=True,
            )

    # make a sparse matrix in adata.obsp['scanit-graph']
    if spatial_graph_kwargs is None:
        spatial_graph_kwargs = {
            'method': 'alpha shape', 
            'alpha_n_layer': 2, 
            'knn_n_neighbors': 5,
        }
    scanit.tl.spatial_graph(adata, **spatial_graph_kwargs)
    # make a N x n_h feature matrix in adata.obsm['X_scanit']
    if spatial_representation is None:
        spatial_representation = {
            'n_h': 30,
            'n_epoch': 2000, 
            'lr': 0.001, 
            'device': 'cuda', 
            'n_consensus': 1, 
            'projection': 'mds', 
            'python_seed': 0, 
            'torch_seed': 0, 
            'numpy_seed': 0,
        }
    scanit.tl.spatial_representation(adata, **spatial_representation)
    
    if return_anndata:
        return adata
    else:
        colnames = [f'scanit_{i}' for i in range(adata.obsm['X_scanit'].shape[1])]
        scanit_features = pd.DataFrame(adata.obsm['X_scanit'], columns=colnames)
        return scanit_features


def make_niches_HMRF(
    X: np.array = None, 
    coords: np.array = None, 
    pairs: np.array = None, 
    var_names: Union[Iterable[str], None] = None,
    k: int = 10,
    betas: list[int] = None,
    ) -> pd.DataFrame:
    """
    Compute niche IDs vector of each node in a network
    given the  hidden Markov random field (HMRF) method
    from the Giotto R package.

    Parameters
    ----------
    X : array_like
        Nodes' attributes on which features are computed.
    coords : array_like
        Coordinates of cells.
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    var_names : list
        Names of variables of X.
    k : int, 10
        Number of niches to find.
    betas : list[int]
        beta value of the HMRF model, controlling the smoothness of
        clustering. If None default values are used based on feature
        numbers, otherwise, a vector of three values: initial beta, 
        beta increment, and number of betas

    Returns
    -------
    feats : dataframe
        Features computed with the HMRF method.
    """

    if betas is None:
        betas = 'NULL'


def compute_spatial_omic_features_single_network(
    method: str = 'NAS',
    net_dir: Union[str, Path] = None,  
    nodes_dir: Union[str, Path] = None,  
    edges_dir: Union[str, Path] = None, 
    data_info: List[str] = None,
    extension: str = None,
    read_fct: Callable = None,
    id_level_1: str = None,
    id_level_2: Union[str, None] = None, 
    col_coords: Union[Iterable, None] = None,
    attributes_col: Union[Iterable, None] = None,
    use_attributes: Union[Iterable, None] = None, 
    make_onehot: bool = False,
    order: int = 1, 
    stat_funcs: Union[str, List[Callable]] = 'default', 
    stat_names: Union[str, List[str]] = 'default', 
    var_sep: str = ' ',
    save_intermediate_results: bool = False, 
    dir_save_interm: Union[str, Path, None] = None,
    add_sample_info: bool = True,
    verbose: int = 1,
    ) -> pd.DataFrame:
    """
    Compute the spatial omic features for a single network.

    Parameters
    ----------
    method : str = 'NAS'
        Method used to compute features from spatial omic data.
        Currently implemented methods are 'NAS' for the Neighbors 
        Aggregation Statistics and 'SCAN-IT'.
    net_dir : Union[str, Path], None
        Directory where network files are stored.
    data_info : List[str, str], None
        Identifier IDs of sample, e.g. patient id and sample id.
    extension : str, None
        File format of network files.
    read_fct : Callable, None
        Function used to load network files.
    id_level_1 : str, None
        Identifier of the first level of the dataset, like
        'patient' or 'chromosome'.
    id_level_2 : Union[str, None], None
        Identifier of the second level of the dataset, like 
        'sample' or 'locus'.
    col_coords : Union[Iterable, None], None
        Coordinate columns if needed by the spatial omic method.
    attributes_col : Union[Iterable, None], None
        Unique columns storing attributes, like cell types, or
        list of columns used to aggregate variables. 
        If None, all columns are used.
    use_attributes : Union[Iterable, None], None
        If provided, subset of variables used for aggregation.
    make_onehot : bool, False
        If True, convert a single column into multiple columns.
    order : int, 1 
        Maximum order of neighborhoud for aggregation.
    stat_funcs : Union[str, List[Callable]], 'default'
        Statistics functions to use on aggregated data. 
        If 'default' np.mean and np.std are use.
        All functions are used with the `axis=0` argument.
    stat_names : Union[str, List[str]], 'default' 
        Names of the statistical functions used on aggregated data.
        If 'default' 'mean' and 'std' are used.
    var_sep : str, ' '
        Separation between variables names and statistical functions names.
    save_intermediate_results : bool, False 
        If True, save results for each network.
    dir_save_interm : Union[str, Path, None], None
        Directory of intermediate results.
    add_sample_info : bool, True
        If True, add sample information to the final NAS table.
    verbose : int, 1
        Level of information displayed.
    
    Returns
    -------
    feats : pd.DataFrame
        Table of spatial omic features.
    """
    assert method in ['NAS', 'SCAN-IT']

    if net_dir is not None:
        net_dir = Path(net_dir)
        if nodes_dir is None:
            nodes_dir = net_dir
        if edges_dir is None:
            edges_dir = net_dir
    nodes_dir = Path(nodes_dir)
    edges_dir = Path(edges_dir)

    # load nodes and edges of a specific group
    if len(data_info) == 1:
        str_group = f'{id_level_1}-{data_info[0]}'
    elif len(data_info) == 2:
        str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
    nodes = read_fct(nodes_dir / f'nodes_{str_group}.{extension}')
    edges = read_fct(edges_dir / f'edges_{str_group}.{extension}')

    if attributes_col is None:
        attributes_col = nodes.columns

    # make dummy variables for attributes (ex: phenotype) if needed
    if make_onehot:
        assert len(attributes_col) == 1, "`attributes_col` has to be of length 1 to make dummy variables"
        nodes = nodes.join(pd.get_dummies(nodes[attributes_col], prefix='', prefix_sep=''))
    if use_attributes is None:
        if len(attributes_col) == 1:
            use_attributes = np.unique(nodes[attributes_col])
        else:
            use_attributes = attributes_col
    else:
        # handle missing columns
        if len(attributes_col) == 1:
           missing_cols = set(use_attributes).difference(np.unique(nodes[attributes_col]))
        else:
           missing_cols = set(use_attributes).difference(np.unique(attributes_col))
        for col in missing_cols:
            nodes[col] = 0
    
    if method == 'NAS':
        # compute Neighbors Aggregation Statistics
        feats = make_features_NAS(
            X=nodes[use_attributes].astype(float).values, 
            pairs=edges.values, 
            order=order, 
            var_names=use_attributes, 
            stat_funcs=stat_funcs, 
            stat_names=stat_names, 
            var_sep=var_sep)
    elif method == 'SCAN-IT':
        if col_coords is None:
            col_coords = ['y', 'x']
        feats = make_features_SCANIT(
            X=nodes[use_attributes].astype(float).values, 
            pairs=edges.values, 
            coords=nodes[col_coords].values, 
            var_names=use_attributes, 
            )
    if add_sample_info:
        feats[id_level_1] = data_info[0]
        feats[id_level_2] = data_info[1]
    
    if save_intermediate_results:
        if dir_save_interm is None:
            dir_save_interm = net_dir / '.temp'
        dir_save_interm.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(dir_save_interm / f'{method}_{str_group}.parquet', index=False)
    
    return feats


def compute_spatial_omic_features_all_networks(
    method: str = 'NAS',
    net_dir: Union[str, Path] = None,  
    nodes_dir: Union[str, Path] = None,  
    edges_dir: Union[str, Path] = None,  
    attributes_col: Union[str, Iterable, None] = None,
    use_attributes: Union[Iterable, None] = None,  
    make_onehot: bool = False,
    stat_funcs: Union[str, List[Callable]] = 'default', 
    stat_names: Union[str, List[str]] = 'default', 
    order: int = 1, 
    id_level_1: str = 'patient',
    id_level_2: Union[str, None] = 'sample',
    extension: str = 'parquet',
    data_index: Union[List[Tuple], None]=None,
    parallel_groups: Union[bool, int, str] = 'max', 
    memory_limit: str = 'max',
    save_intermediate_results: bool = False, 
    dir_save_interm: Union[str, Path, None] = None,
    add_sample_info: bool = True,
    verbose: int = 1,
    ) -> pd.DataFrame:
    """
    Compute the spatial omic features for all
    samples in a batch, cohort or other kind of groups.

    Parameters
    ----------
    method : str = 'NAS'
        Method used to compute features from spatial omic data.
        Currently implemented methods are 'NAS' for the Neighbors 
        Aggregation Statistics and 'SCAN-IT'.
    net_dir : Union[str, Path], None
        Directory where network files are stored.
    attributes_col : Union[str, Iterable, None], None
        Unique columns storing attributes, like cell types, or
        list of columns used to aggregate variables. 
        If None, all columns are used.
    use_attributes : Union[Iterable, None], None
        If provided, subset of variables used for aggregation.
    make_onehot : bool, False
        If True, convert a single column into multiple columns.
    order : int, 1 
        Maximum order of neighborhood for aggregation.
    id_level_1 : str
        Identifier of the first level of the dataset, like
        'patient' or 'chromosome'.
    id_level_2 : Union[str, None], None
        Identifier of the second level of the dataset, like 
        'sample' or 'locus'.
    extension : str
        File format of network files.
    data_index : Union[List[Tuple], None], None
        List of identifier IDs of network files.
    parallel_groups : Union[bool, int, str], 'max'
        Computation is run on a single CPU if False, on the specified 
        number of CPU if it is an integer, or on the max or max-1 
        number of CPUS if it is a string.
    memory_limit : str, 'max'
        Maximum memory used by Dask during parallel computation.
        Use either 'max' or 'XX GB'.
    
    Returns
    -------
    nas : pd.DataFrame
        Table of Neighbors Aggregated Statistics.

    Notes
    -----
    The ordering of observations (cells) in the resulting table may differ
    from the ordering in the original data if cell are not ordered per sample
    or if parallel computation is used.
    """

    if net_dir is not None:
        net_dir = Path(net_dir)
        if nodes_dir is None:
            nodes_dir = net_dir
        if edges_dir is None:
            edges_dir = net_dir
    nodes_dir = Path(nodes_dir)
    edges_dir = Path(edges_dir)
        
    data_single_level = id_level_2 is None
    
    if isinstance(attributes_col, str):
        attributes_col = [attributes_col]
    read_fct = io.get_reader(extension)  # fast Polars-backed reader (mosna.io)
    
    # build index of patients and samples files
    if data_index is None:
        data_index = make_data_index(
            nodes_dir,
            id_level_1,
            id_level_2, 
            extension,
            )
    
    groups_data = []
    
    # redefine defaults values of the network analysis function
    use_compute_sof_single_network = partial(
        compute_spatial_omic_features_single_network,
        method=method,
        net_dir=net_dir,
        nodes_dir=nodes_dir,
        edges_dir=edges_dir,
        extension=extension,
        read_fct=read_fct,
        attributes_col=attributes_col,
        use_attributes=use_attributes, 
        make_onehot=make_onehot,
        stat_funcs=stat_funcs,
        stat_names=stat_names,
        order=order,
        id_level_1=id_level_1,
        id_level_2=id_level_2,
        save_intermediate_results=save_intermediate_results,
        dir_save_interm=dir_save_interm,
        add_sample_info=add_sample_info,
        verbose=0)  # don't display iterations
    
    if parallel_groups is False:
        if verbose > 0:
            iterable = tqdm(data_index, desc='data')
        else:
            iterable = data_index
        for data_info in iterable:
            group_data = use_compute_sof_single_network(data_info=data_info)
            groups_data.append(group_data)
        nas = pd.concat(groups_data, axis=0)
    else:
        from multiprocessing import cpu_count
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        
        # select the right number of cores
        nb_cores = cpu_count()
        if isinstance(parallel_groups, int):
            use_cores = min(parallel_groups, nb_cores)
        elif parallel_groups == 'max-1':
            use_cores = nb_cores - 1
        elif parallel_groups == 'max':
            use_cores = nb_cores
        if memory_limit == 'max':
            total_memory, used_memory, free_memory = map(
                int, os.popen('free -t -m').readlines()[-1].split()[1:])
            memory_limit = str(int(0.95 * free_memory/1000)) + 'GB'

        # set up cluster and workers
        with LocalCluster(
            n_workers=use_cores,
            processes=True,
            threads_per_worker=1,
            memory_limit=memory_limit,
            ) as cluster, Client(cluster) as client:
                # TODO: add dask's progressbar
                for data_info in data_index:
                    # select nodes and edges of a specific group
                    group_data = delayed(use_compute_sof_single_network)(data_info=data_info)
                    groups_data.append(group_data)
                # evaluate the parallel computation
                # ProgressBar().register()
                nas = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()

    return nas


def surv_col_to_numpy(df_surv, event_col, duration_col):
    y_df = df_surv[[event_col, duration_col]].copy()
    y_df.loc[:, event_col] = y_df.loc[:, event_col].astype(bool)
    records = y_df.to_records(index=False)
    y = np.array(records, dtype = records.dtype.descr)
    return y
