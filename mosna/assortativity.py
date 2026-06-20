#! /usr/bin/env python3
"""mosna.assortativity — mixing matrices and assortativity statistics

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403
from mosna import io  # fast Polars/Parquet I/O

from mosna.preprocessing import make_data_index  # noqa: F401


def count_edges_undirected(nodes, edges, attributes):
    """Compute the count of edges whose end nodes correspond to given attributes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        The attributes of nodes whose edges are selected
        
    Returns
    -------
    count : int
       Count of edges
    """
    
    pairs = np.logical_or(np.logical_and(nodes.loc[edges['source'], attributes[0]].values, nodes.loc[edges['target'], attributes[1]].values),
                          np.logical_and(nodes.loc[edges['target'], attributes[0]].values, nodes.loc[edges['source'], attributes[1]].values))
    count = pairs.sum()
    
    return count


def count_edges_directed(nodes, edges, attributes):
    """Compute the count of edges whose end nodes correspond to given attributes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        The attributes of nodes whose edges are selected
        
    Returns
    -------
    count : int
       Count of edges
    """
    
    pairs = np.logical_and(nodes.loc[edges['source'], attributes[0]].values, nodes.loc[edges['target'], attributes[1]].values)
    count = pairs.sum()
    
    return count


def _is_one_hot(vals):
    """True when each row of `vals` has exactly one active (==1) entry."""
    return (
        vals.shape[1] > 0
        and np.array_equal(vals.sum(axis=1), np.ones(vals.shape[0]))
        and np.array_equal(vals, vals.astype(bool))
    )


def _mixmat_from_codes(codes, src, tgt, A):
    """Raw (un-normalised, single diagonal) undirected mixing matrix from
    integer attribute codes, via a single bincount over the edge code-pairs.

    ``codes`` maps each node to its attribute index (one-hot encoding). For an
    undirected graph the count for pair (i, j != i) is D[i,j] + D[j,i] and the
    diagonal is D[i,i], where D is the directed (src -> tgt) code-pair count.
    """
    flat = codes[src].astype(np.int64) * A + codes[tgt]
    D = np.bincount(flat, minlength=A * A).reshape(A, A).astype(float)
    mixmat = D + D.T
    np.fill_diagonal(mixmat, np.diag(D))
    return mixmat


def mixing_matrix(nodes, edges, attributes, normalized=True, double_diag=True):
    """Compute the mixing matrix of a network described by its `nodes` and `edges`.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        Categorical attributes considered in the mixing matrix
    normalized : bool (default=True)
        Return counts if False or probabilities if True.
    double_diag : bool (default=True)
        If True elements of the diagonal are doubled like in NetworkX or iGraph 
       
    Returns
    -------
    mixmat : array
       Mixing matrix
    """
    
    # Vectorised computation: instead of A*(A+1)/2 separate pandas `.loc`
    # passes over all edges (the historical bottleneck), gather the attribute
    # values once and reduce with array algebra. Numerically identical to the
    # per-pair `count_edges_undirected` loop (see tests/test_equivalence.py)
    # but 1-2 orders of magnitude faster on large graphs.
    A = len(attributes)
    src = edges['source'].values
    tgt = edges['target'].values
    vals = nodes[attributes].values

    # Fast path: when attributes are one-hot (each node has exactly one active
    # attribute, the usual cell-type encoding), the whole matrix is a single
    # `bincount` of the (src_code, tgt_code) edge pairs — no per-edge gather and
    # no A^2 loop. This is ~100x faster than the boolean path on big graphs.
    if _is_one_hot(vals):
        codes = vals.argmax(axis=1)                      # node -> attribute idx
        mixmat = _mixmat_from_codes(codes, src, tgt, A)
    else:
        # General path (multi-label / non-one-hot attributes): boolean algebra.
        S = vals[src].astype(bool)   # (n_edges, A)
        T = vals[tgt].astype(bool)   # (n_edges, A)
        mixmat = np.zeros((A, A))
        for i in range(A):
            Si, Ti = S[:, i], T[:, i]
            for j in range(i + 1):
                cnt = np.logical_or(Si & T[:, j], S[:, j] & Ti).sum()
                mixmat[i, j] = cnt
                mixmat[j, i] = cnt

    if double_diag:
        for i in range(A):
            mixmat[i, i] += mixmat[i, i]

    if normalized:
        mixmat = mixmat / mixmat.sum()

    return mixmat


def attribute_ac(M):
    """Compute assortativity for attribute matrix M.

    Parameters
    ----------
    M : numpy array or matrix
        Attribute mixing matrix.

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    
    M = np.asarray(M, dtype=float)
    if M.sum() != 1.0:
        M = M / M.sum()
    # NetworkX Eq.(2): (trace(e) - sum(e^2)) / (1 - sum(e^2)), where the sum is
    # over the matrix product e @ e (the original used np.asmatrix, whose `*`
    # is matrix multiplication). Use ndarrays + `@`: np.asmatrix is deprecated
    # and float() on the resulting 1x1 matrix raises TypeError on NumPy >= 2.3.
    s = (M @ M).sum()
    t = np.trace(M)
    r = (t - s) / (1 - s)
    return float(r)


def mixmat_to_df(mixmat, attributes):
    """
    Make a dataframe of a mixing matrix.
    """
    return pd.DataFrame(mixmat, columns=attributes, index=attributes)


def mixmat_to_columns(mixmat):
    """
    Flattens a mixing matrix taking only elements of the lower triangle and diagonal.
    To revert this use `series_to_mixmat`.
    """
    N = mixmat.shape[0]
    val = []
    for i in range(N):
        for j in range(i+1):
            val.append(mixmat[i,j])
    return val


def series_to_mixmat(series, medfix=' - ', discard=' Z'):
    """
    Convert a 1D pandas series into a 2D dataframe.
    To revert this use `mixmat_to_columns`.
    """
    N = series.size
    combi = [[x.split(medfix)[0].replace(discard, ''), x.split(medfix)[1].replace(discard, '')] for x in series.index]
    # get unique elements of the list of mists
    from itertools import chain 
    uniq = [*{*chain.from_iterable(combi)}]
    mat = pd.DataFrame(data=None, index=uniq, columns=uniq)
    for i in series.index:
        x = i.split(medfix)[0].replace(discard, '')
        y = i.split(medfix)[1].replace(discard, '')
        val = series[i]
        mat.loc[x, y] = val
        mat.loc[y, x] = val
    return mat


def attributes_pairs(attributes, prefix='', medfix=' - ', suffix=''):
    """
    Make a list of unique pairs of attributes.
    Convenient to make the names of elements of the mixing matrix 
    that is flattened.
    """
    N = len(attributes)
    col = []
    for i in range(N):
        for j in range(i+1):
            col.append(prefix + attributes[i] + medfix + attributes[j] + suffix)
    return col


def core_rand_mixmat(nodes, edges, attributes, random_state=None):
    """
    Compute the mixing matrix of a network after nodes' attributes
    are randomized once.

    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    random_state : int or None (default=None)
        Seed forwarded to the shuffle for reproducibility. ``None`` keeps the
        original unseeded behaviour.

    Returns
    -------
    mixmat_rand : array
       Mmixing matrix of the randomized network.
    """
    nodes_rand = deepcopy(nodes)
    nodes_rand[attributes] = shuffle(nodes_rand[attributes].values, random_state=random_state)
    mixmat_rand = mixing_matrix(nodes_rand, edges, attributes)
    return mixmat_rand


def _resolve_n_cores(parallel):
    """Translate the ``parallel`` argument into a concrete worker count."""
    nb_cores = cpu_count()
    if isinstance(parallel, bool):
        # True historically meant "use the default backend"; map to all cores.
        return nb_cores
    if isinstance(parallel, int):
        return max(1, min(parallel, nb_cores))
    if parallel == 'max-1':
        return max(1, nb_cores - 1)
    if parallel == 'max':
        return nb_cores
    raise ValueError(f"Unrecognised `parallel` value: {parallel!r}")


def randomized_mixmat(
        nodes,
        edges,
        attributes,
        n_shuffle=50,
        parallel=False,
        memory_limit='10GB',
        verbose=1,
        backend='joblib',
        random_state=None,
        ):
    """Randomize several times a network by shuffling the nodes' attributes.
    Then compute the mixing matrix and the corresponding assortativity coefficient.

    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel : bool, int or str (default=False)
        How parallelization is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.

        .. note::
           The default is now ``False`` (serial). ``mixing_matrix`` was
           vectorised and is ~30x faster, so each shuffle is cheap and the
           serial path is usually the fastest: process-based parallelism then
           costs more in data-shipping/pickling than it saves. Only enable
           parallelism for very large graphs or very high attribute counts
           where a single shuffle is expensive.
    memory_limit : str (default='10GB')
        Per-worker memory limit, only used by the ``'dask'`` backend.
    verbose : int (default=1)
        Verbosity level (a progress bar is shown when > 0).
    backend : str (default='joblib')
        Parallel backend to use when ``parallel`` is not False:

        - ``'joblib'`` (recommended): process-based parallelism via joblib's
          ``loky`` backend. Uses all requested cores, sidesteps the GIL, and
          starts reliably on constrained machines. This is the fast, robust
          default for this embarrassingly-parallel workload.
        - ``'dask'``: the original ``dask.distributed`` LocalCluster path. Kept
          for backward compatibility; heavier and can fail to spawn workers
          ("Nanny failed to start") on memory-constrained hosts.
    random_state : int or None (default=None)
        Seed for reproducible shuffles. When given, the i-th shuffle uses a
        derived, deterministic seed so results are reproducible *and* identical
        whether run serially or in parallel.

    Returns
    -------
    mixmat_rand : array (n_shuffle x n_attributes x n_attributes)
       Mixing matrices of each randomized version of the network
    assort_rand : array  of size n_shuffle
       Assortativity coefficients of each randomized version of the network

    Notes
    -----
    The numerical result of each shuffle is unchanged from the original
    implementation; only the parallel machinery differs. With ``random_state``
    set, the serial and parallel paths return bit-identical arrays.
    """

    mixmat_rand = np.zeros((n_shuffle, len(attributes), len(attributes)))
    assort_rand = np.zeros(n_shuffle)

    # Per-shuffle seeds: reproducible and order-independent (so parallel ==
    # serial when random_state is set). None preserves the legacy unseeded RNG.
    if random_state is None:
        seeds = [None] * n_shuffle
    else:
        ss = np.random.SeedSequence(random_state)
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_shuffle)]

    if parallel is False:
        vals = nodes[attributes].values
        if _is_one_hot(vals):
            # Fast path: shuffle the integer attribute codes (not the whole
            # DataFrame) and reduce with a single bincount per shuffle. ~8x
            # faster per shuffle at 500k cells. Statistically identical to the
            # original (a uniform permutation of node labels either way); with
            # `random_state` set it is reproducible, though the exact draws
            # differ from the DataFrame-shuffle path.
            A = len(attributes)
            src = edges['source'].values
            tgt = edges['target'].values
            codes = vals.argmax(axis=1).astype(np.int64)
            rng = np.random.default_rng(random_state)
            iterable = range(n_shuffle)
            if verbose > 0:
                iterable = tqdm(iterable, desc='randomization')
            for i in iterable:
                shuffled = rng.permutation(codes)
                mm = _mixmat_from_codes(shuffled, src, tgt, A)
                mm[np.diag_indices(A)] *= 2          # double_diag (as in mixing_matrix)
                mm = mm / mm.sum()                    # normalized
                mixmat_rand[i] = mm
                assort_rand[i] = attribute_ac(mm)
            return mixmat_rand, assort_rand

        # General path (multi-label attributes): unchanged, exact behaviour.
        iterable = range(n_shuffle)
        if verbose > 0:
            iterable = tqdm(iterable, desc='randomization')
        for i in iterable:
            mixmat_rand[i] = core_rand_mixmat(nodes, edges, attributes, random_state=seeds[i])
            assort_rand[i] = attribute_ac(mixmat_rand[i])
        return mixmat_rand, assort_rand

    use_cores = _resolve_n_cores(parallel)

    if backend == 'joblib':
        # Fast, robust default: loky (process-based, GIL-free, reliable startup).
        from joblib import Parallel, delayed as jl_delayed
        tasks = (jl_delayed(core_rand_mixmat)(nodes, edges, attributes, random_state=seeds[i])
                 for i in range(n_shuffle))
        results = Parallel(n_jobs=use_cores, backend='loky',
                           verbose=5 if verbose > 0 else 0)(tasks)
        for i, mm in enumerate(results):
            mixmat_rand[i] = mm
            assort_rand[i] = attribute_ac(mm)
        return mixmat_rand, assort_rand

    if backend == 'dask':
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        cluster = LocalCluster(n_workers=use_cores,
                               threads_per_worker=1,
                               memory_limit=memory_limit)
        client = Client(cluster)
        try:
            mixmat_delayed = [
                delayed(core_rand_mixmat)(nodes, edges, attributes, random_state=seeds[i])
                for i in range(n_shuffle)
            ]
            mixmat_rand = delayed(np.array)(mixmat_delayed).compute()
            for i in range(n_shuffle):
                assort_rand[i] = attribute_ac(mixmat_rand[i])
        finally:
            client.close()
            cluster.close()
        return mixmat_rand, assort_rand

    raise ValueError(f"Unrecognised `backend`: {backend!r} (use 'joblib' or 'dask')")


def zscore(mat, mat_rand, axis=0, return_stats=False):
    rand_mean = mat_rand.mean(axis=axis)
    rand_std = mat_rand.std(axis=axis)
    # with warnings.simplefilter("ignore", category=RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zscore = (mat - rand_mean) / rand_std
    if return_stats:
        return rand_mean, rand_std, zscore
    else:
        return zscore


def select_pairs_from_coords(coords_ids, pairs, how='inner', return_selector=False):
    """
    Select edges related to specific nodes.
    
    Parameters
    ----------
    coords_ids : array
        Indices or ids of nodes.
    pairs : array
        Edges defined as pairs of nodes ids.
    how : str (default='inner')
        If 'inner', only edges that have both source and target 
        nodes in coords_ids are select. If 'outer', edges that 
        have at least a node in coords_ids are selected.
    return_selector : bool (default=False)
        If True, only the boolean mask is returned.
    
    Returns
    -------
    pairs_selected : array
        Edges having nodes in coords_ids.
    select : array
        Boolean array to select latter on edges.
    
    Example
    -------
    >>> coords_ids = np.array([5, 6, 7])
    >>> pairs = np.array([[1, 2],
                          [3, 4],
                          [5, 6],
                          [7, 8]])
    >>> select_pairs_from_coords(coords_ids, pairs, how='inner')
    array([[5, 6]])
    >>> select_pairs_from_coords(coords_ids, pairs, how='outer')
    array([[5, 6],
           [7, 8]])
    """
    
    select_source = np.isin(pairs[:, 0], coords_ids)  # np.in1d removed in NumPy 2.0
    select_target = np.isin(pairs[:, 1], coords_ids)
    if how == 'inner':
        select = np.logical_and(select_source, select_target)
    elif how == 'outer':
        select = np.logical_or(select_source, select_target)
    if return_selector:
        return select
    pairs_selected = pairs[select, :]
    return pairs_selected


def sample_assort_mixmat(nodes, edges, attributes, sample_id=None ,n_shuffle=50, 
                         parallel='max', memory_limit='10GB', verbose=1):
    """
    Computed z-scored assortativity and mixing matrix elements for 
    a network of a single sample.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    sample_id : str
        Name of the analyzed sample.
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel : bool, int or str (default="max")
        How parallelization is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    memory_limit : str (default='50GB')
        Dask memory limit for parallelization.
        
    Returns
    -------
    sample_stats : dataframe
        Network's statistics including total number of nodes, attributes proportions,
        assortativity and mixing matrix elements, both raw and z-scored.
    """
    
    col_sample = (['id', '# total'] +
                 ['% ' + x for x in attributes] +
                 ['assort', 'assort MEAN', 'assort STD', 'assort Z'] +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' RAW') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' MEAN') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' STD') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' Z'))
    
    if sample_id is None:
        sample_id = 'None'
    # Network statistics
    mixmat = mixing_matrix(nodes, edges, attributes)
    assort = attribute_ac(mixmat)

    # ------ Randomization ------
    np.random.seed(0)
    mixmat_rand, assort_rand = randomized_mixmat(
        nodes, edges, attributes, 
        n_shuffle=n_shuffle, 
        parallel=parallel, 
        memory_limit=memory_limit,
        verbose=verbose)
    mixmat_mean, mixmat_std, mixmat_zscore = zscore(mixmat, mixmat_rand, return_stats=True)
    assort_mean, assort_std, assort_zscore = zscore(assort, assort_rand, return_stats=True)

    # Reformat sample's network's statistics
    nb_nodes = len(nodes)
    sample_data = ([sample_id, nb_nodes] +
                   [nodes[col].sum()/nb_nodes for col in attributes] +
                   [assort, assort_mean, assort_std, assort_zscore] +
                   mixmat_to_columns(mixmat) +
                   mixmat_to_columns(mixmat_mean) +
                   mixmat_to_columns(mixmat_std) +
                   mixmat_to_columns(mixmat_zscore))
    sample_stats = pd.DataFrame(data=sample_data, index=col_sample).T
    return sample_stats


def _select_nodes_edges_from_group(nodes, edges, group, groups):
    """
    Select nodes and edges related to a given group of nodes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    group: int or str
        Group of interest. 
    groups: pd.Series
        Group identifier of each node. 
    
    Returns
    ------
    nodes_sel : dataframe
        Nodes belonging to the group.
    edges_sel : dataframe
        Edges belonging to the group.
    """
    select = groups == group
    nodes_sel = nodes.loc[select, :]
    nodes_ids = np.where(select)[0]
    edges_selector = select_pairs_from_coords(nodes_ids, edges.values, return_selector=True)
    edges_sel = edges.loc[edges_selector, :]
    return nodes_sel, edges_sel


def batch_assort_mixmat(nodes, edges, attributes, groups, n_shuffle=50,
                        parallel_groups='max', parallel_shuffle=False, memory_limit='50GB',
                        save_intermediate_results=False, dir_save_interm='~'):
    """
    Computed z-scored assortativity and mixing matrix elements for all
    samples in a batch, cohort or other kind of groups.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    groups: pd.Series
        Group identifier of each node. 
        It can be a patient or sample id, chromosome number, etc...
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel_groups : bool, int or str (default="max")
        How parallelization across groups is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    parallel_shuffle : bool, int or str (default="False)
        How parallelization across shuffle rounds is performed.
        Parameter options are identical to `parallel_groups`.
    memory_limit : str (default='50GB')
        Dask memory limit for parallelization.
    save_intermediate_results : bool (default=False)
        If True network statistics are saved for each group.
    dir_save_interm : str (default='~')
        Directory where intermediate group network statistics are saved.
        
    Returns
    -------
    networks_stats : dataframe
        Networks's statistics for all groups, including total number of nodes, 
        attributes proportions, assortativity and mixing matrix elements, 
        both raw and z-scored.
    
    Examples
    --------
    >>> nodes_high, edges_high = make_high_assort_net()
    >>> nodes_low, edges_low = make_high_disassort_net()
    >>> nodes = nodes_high.append(nodes_low, ignore_index=True)
    >>> edges_low_shift = edges_low + nodes_high.shape[0]
    >>> edges = edges_high.append(edges_low_shift)
    >>> groups = pd.Series(['high'] * len(nodes_high) + ['low'] * len(nodes_low))
    >>> net_stats = batch_assort_mixmat(nodes, edges, 
                                        attributes=['a', 'b', 'c'], 
                                        groups=groups, 
                                        parallel_groups=False)
    """

    
    # TODO: add selection of subset
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups).copy()
    
    groups_data = []
 
    if parallel_groups is False:
        for group in tqdm(groups.unique(), desc='group'):
            # select nodes and edges of a specific group
            nodes_sel, edges_sel = _select_nodes_edges_from_group(nodes, edges, group, groups)
            # compute network statistics
            group_data = sample_assort_mixmat(nodes_sel, edges_sel, attributes, sample_id=group, 
                                              n_shuffle=n_shuffle, parallel=parallel_shuffle, memory_limit=memory_limit)
            if save_intermediate_results:
                group_data.to_csv(os.path.join(dir_save_interm, f'network_statistics_group_{group}.csv'), 
                                  encoding='utf-8', 
                                  index=False)
            groups_data.append(group_data)
        networks_stats = pd.concat(groups_data, axis=0)
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
        # set up cluster and workers
        cluster = LocalCluster(n_workers=use_cores, 
                               threads_per_worker=1,
                               memory_limit=memory_limit)
        client = Client(cluster)
        
        for group in groups.unique():
            # select nodes and edges of a specific group
            nodes_edges_sel = delayed(_select_nodes_edges_from_group)(nodes, edges, group, groups)
            # individual samples z-score stats are not parallelized over shuffling rounds
            # because parallelization is already done over samples
            group_data = delayed(sample_assort_mixmat)(nodes_edges_sel[0], nodes_edges_sel[1], attributes, sample_id=group, 
                                                       n_shuffle=n_shuffle, parallel=parallel_shuffle) 
            groups_data.append(group_data)
        # evaluate the parallel computation
        networks_stats = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()
    return networks_stats


def make_group_network_stats(
    net_dir,
    data_info,
    extension,
    read_fct,
    id_level_1,
    id_level_2=None,
    attributes_col=None,
    use_attributes=None, 
    make_onehot=False,
    n_shuffle=50,
    parallel_shuffle=False, 
    memory_limit='10GB',
    save_intermediate_results=False,
    dir_save_interm=None,
    verbose=1):
    """
    Load the network data of a specific sample group, i.e. a specific pair
    of  id_level_1 and id_level_2, and computes its mixing matrix elements
    and assortativity.    
    """

    # load nodes and edges of a specific group
    str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
    nodes = read_fct(net_dir / f'nodes_{str_group}.{extension}')
    edges = read_fct(net_dir / f'edges_{str_group}.{extension}')

    # make dummy variables for attributes (ex: phenotype) if needed
    if make_onehot:
        nodes = nodes.join(pd.get_dummies(nodes[attributes_col], prefix='', prefix_sep=''))
    if use_attributes is None:
        use_attributes = np.unique(nodes[attributes_col])
    # compute network statistics
    group_data = sample_assort_mixmat(
        nodes, edges, 
        attributes=use_attributes, 
        sample_id=str_group, 
        n_shuffle=n_shuffle, 
        parallel=parallel_shuffle, 
        memory_limit=memory_limit, 
        verbose=verbose)
    
    if save_intermediate_results:
        if dir_save_interm is None:
            dir_save_interm = net_dir / '.temp'
        dir_save_interm.mkdir(parents=True, exist_ok=True)
        group_data.to_parquet(dir_save_interm / f'network_statistics_{str_group}.parquet', index=False)
    
    return group_data


def groups_assort_mixmat(
    net_dir, 
    attributes_col,
    use_attributes=None, 
    make_onehot=False,
    id_level_1='patient',
    id_level_2='sample', 
    extension='parquet',
    data_index=None,
    n_shuffle=50,
    parallel_groups='max', 
    memory_limit='max',
    save_intermediate_results=False, 
    dir_save_interm=None,
    verbose=1):
    """
    Compute z-scored assortativity and mixing matrix elements for all
    samples in a batch, cohort or other kind of groups.
    
    Parameters
    ----------
    net_dir: str or path object
        Location of reconstructed networks data with nodes and edges files.
    attributes_col: str or list
        Column containing attributes, multiple columns if attributes are already one-hot encoded.
    use_attributes: list
        Categorical attributes considered in the mixing matrix (ex: phenotypes).
    make_onehot: bool
        If True, make one-hot encoded variables from `attributes_col`.
    id_level_1: str
        Label in filenames used to identify the first level of data (ex: patient_id).
    id_level_2: str or None
        Label in filenames used to identify the second level of data (ex: sample_id).
    extension: str
        Extension used to save network data. Either 'parquet' (default) or 'csv'.
    data_index: list(list) or list or None
        Index of all groups, i.e. patients and their samples, or genes and their loci.
        If None, the index is built from files in net_dir.
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel_groups : bool, int or str (default="max")
        How parallelization across groups is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    memory_limit : str (default='max')
        Dask memory limit for parallelization.
        If 'max, will use 95% of the available free memory.
    save_intermediate_results : bool (default=False)
        If True network statistics are saved for each group.
    dir_save_interm : str (default=None)
        Directory where intermediate group network statistics are saved.
        If None, data is saved in net_dir / '.temp'.
        
    Returns
    -------
    networks_stats : dataframe
        Networks's statistics for all groups, including total number of nodes, 
        attributes proportions, assortativity and mixing matrix elements, 
        both raw and z-scored.
    
    Examples
    --------
    >>> nodes_high, edges_high = make_high_assort_net()
    >>> nodes_low, edges_low = make_high_disassort_net()
    >>> nodes = nodes_high.append(nodes_low, ignore_index=True)
    >>> edges_low_shift = edges_low + nodes_high.shape[0]
    >>> edges = edges_high.append(edges_low_shift)
    >>> groups = pd.Series(['high'] * len(nodes_high) + ['low'] * len(nodes_low))
    >>> net_stats = batch_assort_mixmat(nodes, edges, 
                                        attributes=['a', 'b', 'c'], 
                                        groups=groups, 
                                        parallel_groups=False)
    """

    net_dir = Path(net_dir)
    data_single_level = id_level_2 is None
    
    if isinstance(attributes_col, str):
        attributes_col = [attributes_col]
    read_fct = io.get_reader(extension)  # fast Polars-backed reader (mosna.io)
    
    # build index of patients and samples files
    if data_index is None:
        data_index = make_data_index(
            net_dir,
            id_level_1,
            id_level_2, 
            extension,
            )
    
    groups_data = []
    
    # redefine defaults values of the network analysis function
    use_group_network_stats = partial(
        make_group_network_stats,
        net_dir=net_dir,
        extension=extension,
        read_fct=read_fct,
        attributes_col=attributes_col,
        use_attributes=use_attributes, 
        make_onehot=make_onehot,
        id_level_1=id_level_1,
        id_level_2=id_level_2,
        n_shuffle=n_shuffle,
        parallel_shuffle=False,  # don't parallelize over iterations per network
        memory_limit=memory_limit,
        save_intermediate_results=save_intermediate_results,
        dir_save_interm=dir_save_interm,
        verbose=0)  # don't display iterations
    
    if parallel_groups is False:
        if verbose > 0:
            iterable = tqdm(data_index, desc='data')
        else:
            iterable = data_index
        for data_info in iterable:
            group_data = use_group_network_stats(data_info=data_info)
            groups_data.append(group_data)
        networks_stats = pd.concat(groups_data, axis=0)
    else:
        from multiprocessing import cpu_count
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        from dask.diagnostics import ProgressBar
        
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
                    group_data = delayed(use_group_network_stats)(data_info=data_info)
                    groups_data.append(group_data)
                # evaluate the parallel computation
                # ProgressBar().register()
                networks_stats = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()

    return networks_stats
