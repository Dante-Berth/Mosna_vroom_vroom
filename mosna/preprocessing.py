#! /usr/bin/env python3
"""mosna.preprocessing — data transformation, aggregation and batch correction

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403
from mosna import io  # fast Polars/Parquet I/O


def transform_CLR(X: np.ndarray):
    X = X.copy()
    X[X == 0] = X.max() / 100000
    X_out = cs.clr(cs.closure(X))
    return X_out


def transform_logp1(X):
    return np.log(X + 1)


def transform_data(
    data, 
    groups=None,
    use_cols=None,
    method='clr'):
    """
    Perform data transformation

    Parameters
    ----------
    data : ndarray or DataFrame
        Data to transform.
    groups : Iterable
        List of group (batch, sample, etc...).
    use_cols : Iterable, None
        List of columns to use if data is a DataFrame.
    method : str
        Method of data transformation.
    
    Returns
    -------
    data_out : ndarray
        Transformed data.
    """
    if not isinstance(data, np.ndarray):
        # data is a DataFrame, recursively call this function on the 
        # exracted values as an ndarray
        data_out = data.copy()
        if groups is not None and isinstance(groups, str):
            # groups is a varibale name, extract a list
            groups = data[groups]
        if use_cols is None:
            use_cols = data.columns
        data_out.loc[:, use_cols] = transform_data(
            data_out.loc[:, use_cols].values, 
            groups=groups,
            method=method)
    else:
        # data is an ndarray
        if method == 'clr':
            fct_transfo = transform_CLR
        elif method == 'logp1':
            fct_transfo = transform_logp1
        
        if groups is None:
            data_out = fct_transfo(data)
        else:
            data_out = data.copy()
            for group in np.unique(groups):
                select = groups == group
                data_out[select] = fct_transfo(data_out[select])
    return data_out


def make_data_index(
    nodes_dir: Union[str, Path],
    id_level_1: str = 'patient',
    id_level_2: Union[str, None] = 'sample', 
    extension: str = 'parquet',
    ):
    """
    Make an index of patient and samples ids.
    """

    data_index = []
    len_ext = len(extension) + 1
    len_l1 = len(id_level_1) + 1
    files = nodes_dir.glob(f'nodes_*.{extension}')
    data_single_level = id_level_2 is None

    if data_single_level:
        for file in files:
            # parse patient and sample description
            file_name = file.name[6:-len_ext]
            patient_info = file_name.split('_')[0]
            patient_id = patient_info[len_l1:]
            
            # add info to data index
            data_index.append([patient_id])
    else:
        len_l2 = len(id_level_2) + 1
        for file in files:
            # parse patient and sample description
            file_name = file.name[6:-len_ext]
            patient_info, sample_info = file_name.split('_')
            patient_id = patient_info[len_l1:]
            sample_id = sample_info[len_l2:]
            
            # add info to data index
            data_index.append((patient_id, sample_id))

    return data_index


def transform_nodes(
    nodes_dir: Union[str, Path],
    id_level_1: str = 'patient',
    id_level_2: Union[str, None] = 'sample', 
    extension: str = 'parquet',
    data_index: Union[List[Tuple], None] = None,
    use_cols: Union[Iterable, None] = None,
    method: str = 'clr',
    save_dir: Union[str, Path] = 'auto',
    force_recompute: bool = False,
    ):
    """
    Load nodes data in a directory, transform and save them
    in a sub-directory.

    Parameters
    ----------
    nodes_dir : Union[Path, str]
        Nodes directory.
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
    use_cols : Iterable, None
        List of columns to use if data is a DataFrame.
    method : str
        Data transformation method.
    save_dir : Union[Path, str, None]
        If auto, save_dir is a sub-folder of nodes_dir named after
        the data transformation method.
    force_recompute : bool, False
        If True, recompute and rewrite output even if it
        already exists on disk.
    
    Returns
    -------
    save_dir : Path
        Final save directory.
    """

    nodes_dir = Path(nodes_dir)
    data_single_level = id_level_2 is None

    read_fct = io.get_reader(extension)  # fast Polars-backed reader (mosna.io)

    # build index of patients and samples files
    if data_index is None:
        data_index = make_data_index(
            nodes_dir,
            id_level_1,
            id_level_2, 
            extension,
            )

    if save_dir == 'auto':
        save_dir = nodes_dir / f"transfo-{method}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for data_info in data_index:
        # load nodes of a specific group
        if len(data_info) == 1:
            str_group = f'{id_level_1}-{data_info[0]}'
        elif len(data_info) == 2:
            str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
        file_name = save_dir / f'nodes_{str_group}.parquet'
        if not file_name.exists() or force_recompute:
            nodes = read_fct(nodes_dir / f'nodes_{str_group}.{extension}')

            nodes_transfo = transform_data(
                data=nodes, 
                groups=None,  # node files already for a single sample or patient
                use_cols=use_cols,
                method=method)

            nodes_transfo.to_parquet(file_name, index=False)
        
    return save_dir


def aggregate_nodes(
    nodes_dir: Union[str, Path],
    id_level_1: str = 'patient',
    id_level_2: Union[str, None] = 'sample', 
    extension: str = 'parquet',
    data_index: Union[List[Tuple], None] = None,
    use_cols: Union[Iterable, None] = None,
    add_sample_info: bool = True,
    ):
    """
    Load nodes data in a directory, aggregate them and return
    or save them.

    Parameters
    ----------
    nodes_dir : Union[Path, str]
        Nodes directory.
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
    use_cols : Iterable, None
        List of columns to use if data is a DataFrame.
    method : str
        Data transformation method.
    add_sample_info : bool, True
        If True, add sample information to nodes data.
    
    Returns
    -------
    nodes_agg : Union[None, pd.DataFrame]
        Aggregated nodes data.
    """

    nodes_dir = Path(nodes_dir)
    data_single_level = id_level_2 is None

    read_fct = io.get_reader(extension)  # fast Polars-backed reader (mosna.io)

    # build index of patients and samples files
    if data_index is None:
        data_index = make_data_index(
            nodes_dir,
            id_level_1,
            id_level_2, 
            extension,
            )
    
    nodes_agg = []
    for data_info in data_index:
        # load nodes of a specific group
        if len(data_info) == 1:
            str_group = f'{id_level_1}-{data_info[0]}'
        elif len(data_info) == 2:
            str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
        nodes = read_fct(nodes_dir / f'nodes_{str_group}.{extension}')
        if use_cols is not None:
            nodes = nodes[use_cols]
        if add_sample_info:
            nodes[id_level_1] = data_info[0]
            if not data_single_level:
                nodes[id_level_2] = data_info[1]
        nodes_agg.append(nodes)
    
    nodes_agg = pd.concat(nodes_agg, axis=0, ignore_index=True)

    return nodes_agg


def batch_correct_nodes_agg(
    nodes_agg: pd.DataFrame,
    batch_key: str = 'patient',
    use_cols: Union[Iterable, None] = None,
    max_dimred: int = 100,
    return_dense: bool = True,
    add_sample_info: bool = True,
    id_level_1: str = 'patient',
    id_level_2: str = 'sample', 
    ):
    """
    Batch correct omic data in aggregated nodes data with scanorama.

    Parameters
    ----------
    nodes_agg : pd.DataFrame
        Aggregated nodes data.
    batch_key : str, 'patient'
        Batch key used to partition data.
    use_cols : Iterable, None
        List of columns to use.
    max_dimred : int, 100
        Dimensionality used by scanorama for batch correction.
    add_sample_info : bool, True
        If True, add sample information to nodes data.
    return_dense : bool, True
        Return ndarray instead of csr matrix.
    id_level_1 : str, 'patient'
        Identifier of the first level of the dataset, like
        'patient' or 'chromosome'.
    id_level_2 : Union[str, None], None
        Identifier of the second level of the dataset, like 
        'sample' or 'locus'.
    
    Returns
    -------
    nodes_corr : pd.DataFrame
        Batch corrected nodes data.
    """

    # make list of datasets
    datasets = []
    uniq_keys = nodes_agg[batch_key].unique()
    if use_cols is None:
        use_cols = nodes_agg.columns
    datasets = [nodes_agg.loc[nodes_agg[batch_key] == key, use_cols].values for key in uniq_keys]

    # Set variable names for each dataset in datasets
    variable_names = [use_cols for _ in uniq_keys]

    # perform batch correction
    import scanorama
    dimred = min(max_dimred, len(use_cols))
    corrected, _ = scanorama.correct(
        datasets, 
        genes_list=variable_names,
        return_dense=return_dense,
        dimred=dimred,
        )
    
    # make DataFrame
    nodes_corr = pd.DataFrame(np.vstack(corrected), columns=use_cols)
    if add_sample_info:
        nodes_corr[id_level_1] = nodes_agg[id_level_1]
        nodes_corr[id_level_2] = nodes_agg[id_level_2]

    return nodes_corr


def batch_correct_nodes(
    nodes_dir: Union[str, Path],
    id_level_1: str = 'patient',
    id_level_2: str = 'sample', 
    extension: str = 'parquet',
    data_index: Union[List[Tuple], None] = None,
    use_cols: Union[Iterable, None] = None,
    add_sample_info: bool = True,
    batch_key: str = 'patient',
    max_dimred: int = 100,
    return_dense: bool = True,
    save_dir: Union[str, Path] = 'auto',
    force_recompute: bool = False,
    return_nodes: bool = False,
    verbose: int = 0,
    ):
    """
    Batch correct omic data from nodes in a directory.

    Parameters
    ----------
    nodes_dir : Union[Path, str]
        Nodes directory.
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
    use_cols : Iterable, None
        List of columns to use if data is a DataFrame.
    method : str
        Data transformation method.
    add_sample_info : bool, True
        If True, add sample information to nodes data.
    batch_key : str, 'patient'
        Batch key used to partition data.
    max_dimred : int, 100
        Dimensionality used by scanorama for batch correction.
    return_dense : bool, True
        Return ndarray instead of csr matrix.
    save_dir : Union[Path, str]
        If auto, save_dir is a sub-folder of nodes_dir.
    force_recompute : bool, False
        If True, recompute and rewrite output even if it
        already exists on disk.
    return_nodes : bool, False
        Return batch corrected aggregated nodes.
    verbose : int, 0
        Verbosity level.
    
    Returns
    -------
    save_dir : Path
        Final save directory.
    """

    nodes_dir = Path(nodes_dir)
    data_single_level = id_level_2 is None
    
    if save_dir == 'auto':
        save_dir = nodes_dir / f"batch_correction-scanorama_on-{batch_key}"
    save_dir.mkdir(parents=True, exist_ok=True)

    nodes_agg = aggregate_nodes(
        nodes_dir=nodes_dir,
        id_level_1=id_level_1,
        id_level_2=id_level_2, 
        extension=extension,
        data_index=data_index,
        use_cols=None, # aggregate all info (coordinates, markers, ...)
        add_sample_info=add_sample_info,
        )

    if not force_recompute:
        # check if all data already exist
        all_exist = True
        for id_1 in nodes_agg[id_level_1].unique():
            select_1 = nodes_agg[id_level_1] == id_1
            nodes_1 = nodes_agg.loc[select_1, :]
            if data_single_level:
                str_group = f'{id_level_1}-{id_1}'
                file_name = save_dir / f'nodes_{str_group}.parquet'
                if not file_name.exists():
                    all_exist = False
                    break
            else:
                for id_2 in nodes_1[id_level_2].unique():
                    select_2 = nodes_1[id_level_2] == id_2
                    str_group = f'{id_level_1}-{id_1}_{id_level_2}-{id_2}'
                    file_name = save_dir / f'nodes_{str_group}.parquet'
                    if not file_name.exists():
                        all_exist = False
                        break
        if not all_exist:
            force_recompute = True
            if verbose > 0:
                print("Some output files are missing, starting computation of batch correction")
        else:
            if verbose > 0:
                print("All output files already exist, skipping computation of batch correction")
            if return_nodes:
                    nodes_corr = aggregate_nodes(
                        nodes_dir=save_dir,
                        id_level_1=id_level_1,
                        id_level_2=id_level_2, 
                        extension=extension,
                        data_index=data_index,
                        use_cols=None, # aggregate all info (coordinates, markers, ...)
                        add_sample_info=add_sample_info,
                        )

    if force_recompute:
        import scanorama
        nodes_agg_corr = batch_correct_nodes_agg(
            nodes_agg=nodes_agg,
            batch_key=batch_key,
            use_cols=use_cols,
            max_dimred=max_dimred,
            return_dense=return_dense,
            add_sample_info=False, 
            )
        
        # replace raw aggregated variables by batch corrected variables
        # while keeping all other variables
        nodes_corr = nodes_agg.copy()
        nodes_corr.loc[:, use_cols] = nodes_agg_corr.loc[:, use_cols]
        del nodes_agg_corr

        # save nodes data
        for id_1 in nodes_corr[id_level_1].unique():
            select_1 = nodes_corr[id_level_1] == id_1
            nodes_1 = nodes_corr.loc[select_1, :]
            if data_single_level:
                str_group = f'{id_level_1}-{id_1}'
                nodes_1.to_parquet(save_dir / f'nodes_{str_group}.parquet', index=False)
            else:
                for id_2 in nodes_1[id_level_2].unique():
                    select_2 = nodes_1[id_level_2] == id_2
                    nodes_2 = nodes_1.loc[select_2, :]
                    str_group = f'{id_level_1}-{id_1}_{id_level_2}-{id_2}'
                    nodes_2.to_parquet(save_dir / f'nodes_{str_group}.parquet', index=False)

    if return_nodes:
        return save_dir, nodes_corr
    return save_dir
