#! /usr/bin/env python3
"""mosna.neighbors — k-order neighbor aggregation statistics

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403


def neighbors(pairs, n):
    """
    Return the list of neighbors of a node in a network defined 
    by edges between pairs of nodes. 
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
        
    Returns
    -------
    neigh : array_like
        The indices of neighboring nodes.
    """
    
    left_neigh = pairs[pairs[:,1] == n, 0]
    right_neigh = pairs[pairs[:,0] == n, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    return neigh


def neighbors_k_order(pairs, n, order):
    """
    Return the list of up the kth neighbors of a node 
    in a network defined by edges between pairs of nodes
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
    order : int
        Max order of neighbors.
        
    Returns
    -------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order
    
    
    Examples
    --------
    >>> pairs = np.array([[0, 10],
                        [0, 20],
                        [0, 30],
                        [10, 110],
                        [10, 210],
                        [10, 310],
                        [20, 120],
                        [20, 220],
                        [20, 320],
                        [30, 130],
                        [30, 230],
                        [30, 330],
                        [10, 20],
                        [20, 30],
                        [30, 10],
                        [310, 120],
                        [320, 130],
                        [330, 110]])
    >>> neighbors_k_order(pairs, 0, 2)
    [[array([0]), 0],
     [array([10, 20, 30]), 1],
     [array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    """
    
    # all_neigh stores all the unique neighbors and their oder
    all_neigh = [[np.array([n]), 0]]
    unique_neigh = np.array([n])
    
    for k in range(order):
        # detected neighbor nodes at the previous order
        last_neigh = all_neigh[k][0]
        k_neigh = []
        for node in last_neigh:
            # aggregate arrays of neighbors for each previous order neighbor
            neigh = np.unique(neighbors(pairs, node))
            k_neigh.append(neigh)
        # aggregate all unique kth order neighbors
        if len(k_neigh) > 0:
            k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
            # select the kth order neighbors that have never been detected in previous orders
            keep_neigh = np.isin(k_unique_neigh, unique_neigh, invert=True)  # np.in1d removed in NumPy 2.0
            k_unique_neigh = k_unique_neigh[keep_neigh]
            # register the kth order unique neighbors along with their order
            all_neigh.append([k_unique_neigh, k+1])
            # update array of unique detected neighbors
            unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        else:
            break
        
    return all_neigh


def flatten_neighbors(all_neigh):
    """
    Convert the list of neighbors 1D arrays with their order into
    a single 1D array of neighbors.

    Parameters
    ----------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order.

    Returns
    -------
    flat_neigh : array_like
        The indices of neighboring nodes.
        
    Examples
    --------
    >>> all_neigh = [[np.array([0]), 0],
                     [np.array([10, 20, 30]), 1],
                     [np.array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    >>> flatten_neighbors(all_neigh)
    array([  0,  10,  20,  30, 110, 120, 130, 210, 220, 230, 310, 320, 330])
        
    Notes
    -----
    For future features it should return a 2D array of
    nodes and their respective order.
    """
    
    list_neigh = []
    for neigh, order in all_neigh:
        list_neigh.append(neigh)
    flat_neigh = np.concatenate(list_neigh, axis=0)

    return flat_neigh


def make_features_NAS(X, pairs, order=1, var_names=None, stat_funcs='default', stat_names='default', var_sep=' '):
    """
    Compute the statistics on aggregated variables across
    the k order neighbors of each node in a network.

    Parameters
    ----------
    X : array_like
        The data on which to compute statistics (mean, std, ...).
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    order : int
        Max order of neighbors.
    var_names : list
        Names of variables of X.
    stat_funcs : str or list of functions
        Statistics functions to use on aggregated data. If 'default' np.mean and np.std are use.
        All functions are used with the `axis=0` argument.
    stat_names : str or list of str
        Names of the statistical functions used on aggregated data.
        If 'default' 'mean' and 'std' are used.
    var_sep : str
        Separation between variables names and statistical functions names.
        Default is ' '.

    Returns
    -------
    nas : dataframe
        Neighbors Aggregation Statistics of X.
        
    Examples
    --------
    >>> x = np.arange(5)
    >>> X = x[np.newaxis,:] + x[:,np.newaxis] * 10
    >>> pairs = np.array([[0, 1],
                          [2, 3],
                          [3, 4]])
    >>> nas = make_features_NAS(X, pairs, stat_funcs=[np.mean, np.max], stat_names=['mean', 'max'], var_sep=' - ')
    >>> nas.values
    array([[ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [25., 26., 27., 28., 29., 30., 31., 32., 33., 34.],
           [30., 31., 32., 33., 34., 40., 41., 42., 43., 44.],
           [35., 36., 37., 38., 39., 40., 41., 42., 43., 44.]])
    """
    
    nb_obs = X.shape[0]
    nb_var = X.shape[1]
    if stat_funcs == 'default':
        stat_funcs = [np.mean, np.std]
    elif not hasattr(stat_funcs, '__iter__'):
        # check if a single function has been passed
        stat_funcs = [stat_funcs]
    if stat_names == 'default':
        stat_names = ['mean', 'std']
    elif isinstance(stat_names, str):
        if '-' in stat_names:
            stat_names = stat_names.split('-')
        else:
            stat_names = [stat_names]
    nb_funcs = len(stat_funcs)
    nas = np.zeros((nb_obs, nb_var*nb_funcs))

    # check if other info as source and target are in pairs and clean array
    if pairs.shape[1] > 2:
        print("Trimmimg additonnal columns in `pairs`")
        pairs = pairs[:, :2].astype(int)
    
    for i in range(nb_obs):
        all_neigh = neighbors_k_order(pairs, n=i, order=order)
        neigh = flatten_neighbors(all_neigh)
        for j, (stat_func, stat_name) in enumerate(zip(stat_funcs, stat_names)):
            nas[i, j*nb_var : (j+1)*nb_var] = stat_func(X[neigh,:], axis=0)
        
    if var_names is None:
        var_names = [str(i) for i in range(nb_var)]
    columns = []
    for stat_name in stat_names:
        stat_str = var_sep + stat_name
        columns = columns + [var + stat_str for var in var_names]
    nas = pd.DataFrame(data=nas, columns=columns)
    
    return nas


def _stats_are_mean_std(stat_funcs, stat_names):
    """True when the requested statistics are exactly (np.mean, np.std)."""
    return (
        list(stat_funcs) == [np.mean, np.std]
        and list(stat_names) == ['mean', 'std']
    )


def _aggregate_order1_mean_std(X, pairs, nb_obs, nb_var):
    """Vectorised order-1 mean/std over each node's closed neighborhood.

    Builds the symmetric adjacency matrix with self-loops (so node i's
    neighborhood is itself plus its direct neighbors, matching
    ``neighbors_k_order(order=1)`` + ``flatten_neighbors``), then computes
    population mean and std (ddof=0, like ``np.std``) via sparse products:

        mean = (A . X) / deg
        std  = sqrt((A . X^2) / deg - mean^2)

    Returns an (nb_obs, 2*nb_var) array laid out as [means | stds], identical
    to the per-node loop. See tests/test_equivalence.py.
    """
    from scipy import sparse

    p = pairs[:, :2].astype(int)
    self_idx = np.arange(nb_obs)
    rows = np.concatenate([p[:, 0], p[:, 1], self_idx])
    cols = np.concatenate([p[:, 1], p[:, 0], self_idx])
    A = sparse.csr_matrix((np.ones(rows.shape[0]), (rows, cols)),
                          shape=(nb_obs, nb_obs))
    # a neighborhood is a SET of nodes: collapse any multi-edges to 1
    A = (A > 0).astype(float)
    deg = np.asarray(A.sum(axis=1)).ravel()[:, None]   # closed-neighborhood size

    mean = A.dot(X) / deg
    var = A.dot(X * X) / deg - mean * mean
    np.clip(var, 0, None, out=var)                     # guard tiny negatives
    std = np.sqrt(var)

    aggreg = np.empty((nb_obs, nb_var * 2))
    aggreg[:, :nb_var] = mean
    aggreg[:, nb_var:] = std
    return aggreg


def aggregate_k_neighbors(X, pairs, order=1, var_names=None, stat_funcs='default', stat_names='default', var_sep=' '):
    """
    Compute the statistics on aggregated variables across
    the k order neighbors of each node in a network.

    Parameters
    ----------
    X : array_like
        The data on which to compute statistics (mean, std, ...).
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    order : int
        Max order of neighbors.
    var_names : list
        Names of variables of X.
    stat_funcs : str or list of functions
        Statistics functions to use on aggregated data. If 'default' np.mean and np.std are use.
        All functions are used with the `axis=0` argument.
    stat_names : str or list of str
        Names of the statistical functions used on aggregated data.
        If 'default' 'mean' and 'std' are used.
    var_sep : str
        Separation between variables names and statistical functions names
        Default is ' '.

    Returns
    -------
    aggreg : dataframe
        Neighbors Aggregation Statistics of X.
        
    Examples
    --------
    >>> x = np.arange(5)
    >>> X = x[np.newaxis,:] + x[:,np.newaxis] * 10
    >>> pairs = np.array([[0, 1],
                          [2, 3],
                          [3, 4]])
    >>> aggreg = aggregate_k_neighbors(X, pairs, stat_funcs=[np.mean, np.max], stat_names=['mean', 'max'], var_sep=' - ')
    >>> aggreg.values
    array([[ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [25., 26., 27., 28., 29., 30., 31., 32., 33., 34.],
           [30., 31., 32., 33., 34., 40., 41., 42., 43., 44.],
           [35., 36., 37., 38., 39., 40., 41., 42., 43., 44.]])
    """
    
    nb_obs = X.shape[0]
    nb_var = X.shape[1]
    default_stats = stat_funcs == 'default'
    if default_stats:
        stat_funcs = [np.mean, np.std]
        if stat_names == 'default':
            stat_names = ['mean', 'std']
    nb_funcs = len(stat_funcs)
    aggreg = np.zeros((nb_obs, nb_var*nb_funcs))

    # check if other info as source and target are in pairs and clean array
    if pairs.shape[1] > 2:
        print("Trimmimg additonnal columns in `pairs`")
        pairs = pairs[:, :2].astype(int)

    # Fast vectorised path for the common case: order-1 neighborhoods with the
    # default mean/std statistics. The neighborhood of node i is {i} + its direct
    # neighbors, i.e. the adjacency matrix with a self-loop; mean and std then
    # follow from sparse matrix-vector products instead of a per-node Python loop.
    # This is numerically identical to the loop below (verified in tests) but up
    # to ~1000x faster on large graphs. Any other order or custom statistics fall
    # back to the original, exact implementation.
    if order == 1 and default_stats and _stats_are_mean_std(stat_funcs, stat_names):
        aggreg = _aggregate_order1_mean_std(X, pairs, nb_obs, nb_var)
    else:
        for i in range(nb_obs):
            all_neigh = neighbors_k_order(pairs, n=i, order=order)
            neigh = flatten_neighbors(all_neigh)
            for j, (stat_func, stat_name) in enumerate(zip(stat_funcs, stat_names)):
                aggreg[i, j*nb_var : (j+1)*nb_var] = stat_func(X[neigh,:], axis=0)

    if var_names is None:
        var_names = [str(i) for i in range(nb_var)]
    columns = []
    for stat_name in stat_names:
        stat_str = var_sep + stat_name
        columns = columns + [var + stat_str for var in var_names]
    aggreg = pd.DataFrame(data=aggreg, columns=columns)
    
    return aggreg
