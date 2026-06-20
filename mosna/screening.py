#! /usr/bin/env python3
"""mosna.screening — parameter screening for niche detection

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403

from mosna.clustering import get_clusterer, make_niches_composition  # noqa: F401
from mosna.features import surv_col_to_numpy  # noqa: F401
from mosna.modeling import logistic_regression  # noqa: F401
from mosna.plotting import plot_heatmap  # noqa: F401


def screen_nas_parameters(
    status_pred: pd.DataFrame,
    var_aggreg: pd.DataFrame = None,
    var_aggreg_samples_info: pd.DataFrame = None,
    pred_type: str = 'binary',
    predict_key: str = 'sample',
    group_col: str = None,
    var_label: str = None,
    duration_col: str = None,
    event_col: str = None,
    covariates: Iterable = [],
    strata: str = None,
    drop_nan: bool = True,
    split_train_test: bool = True,
    cv_train: int = 5,
    cv_adapt: bool = True, 
    cv_max:int = 10, 
    min_cluster_pred: int = 2,
    max_cluster_pred: int = 200,
    min_score_plot: float = 0.85,
    sof_dir: Union[str, Path] = None,
    dir_save_interm: Union[str, Path] = None,
    iter_reducer_type: Iterable = None,
    iter_dim_clust: Iterable = None,
    iter_n_neighbors: Iterable = None,
    iter_metric: Iterable = None,
    iter_clusterer_type: Iterable = None,
    iter_normalize: Iterable = None,
    clust_size_params: dict = None,
    iter_k_cluster: Iterable = None,
    plot_heatmap: bool = False,
    plot_alphas: bool = False,
    plot_best_model_coefs: bool = False,
    train_model: bool = True,
    recompute: bool = False,
    show_progress: bool = False,
    n_jobs_gridsearch: int = -1,
    verbose: int = 1,
    ):
    """
    Perform grid-search for hyperparameters of the NAS pipeline.

    Parameters
    ----------
    status_pred : pd.DataFrame
        Table containing clinical or biological data for samples or patients.
    var_aggreg : pd.DataFrame = None
        Aggregated statistics of omics data for each cell's neighborhood.
    var_aggreg_samples_info : pd.DataFrame = None
        Sample and patient data for each cell.
    pred_type : str = 'binary'
        Type of prediction task to perform.
    predict_key : str = 'sample'
        Whether predicitons are made per patient or per sample.
    group_col : str = None
        Column of binary value to predict (response, etc...)
    var_label : str = None
        Column of sample or patient IDs.
    duration_col : str = None
        Column of survival duration.
    event_col : str = None
        Column indicating event, like death.
    split_train_test: bool = True
        Set to False to model data, not to predict from it.
    min_score_plot: float = 0.85
        Minimum value of ROC AUC to plot heatmaps.
    """

    assert pred_type in ('binary', 'survival'), "`pred_type` must be 'binary' or 'survival'"
    assert predict_key in ('sample', 'patient'), "`predict_key` must be 'sample' or 'patient'"

    columns = ['dim_clust', 'n_neighbors', 'metric', 'clusterer_type', 
                'k_cluster', 'clust_size_param', 'n_clusters', 'normalize', 
                'l1_ratio', 'alpha']
    col_types = {
        'dim_clust': int,
        'n_neighbors': int,
        'metric': 'category',
        'k_cluster': int,
        'clusterer_type': 'category',
        'clust_size_param': float,
        'n_clusters': int,
        'normalize': 'category',
        'l1_ratio': float,
        'alpha': float,
        }
    l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
    min_alpha = 0.001

    if pred_type == 'binary':
        columns.extend(['score_roc_auc', 'score_ap', 'score_mcc'])
        col_types['score_roc_auc'] = float
        col_types['score_ap'] = float
        col_types['score_mcc'] = float
        if dir_save_interm is None:
            dir_save_interm = sof_dir / f'search_LogReg_on_{predict_key}'
            dir_save_interm.mkdir(parents=True, exist_ok=True)
    elif pred_type == 'survival':
        columns.extend(['n_coeffs', 'score'])
        col_types['n_coeffs'] = int
        col_types['score'] = float
        if dir_save_interm is None:
            dir_save_interm = sof_dir / f'search_CoxPH_on_{predict_key}'
            dir_save_interm.mkdir(parents=True, exist_ok=True)

    aggregated_path = dir_save_interm / 'all_models.parquet'
    if aggregated_path.exists() and not recompute:
        if verbose > 0:
            print('Load NAS hyperparameters search results')
        all_models = pd.read_parquet(aggregated_path)
        return all_models
    elif not aggregated_path.exists() and not recompute:
        if verbose > 0:
            print('Aggregate NAS hyperparameters search results')
        all_models = [pd.read_parquet(file_path) for file_path in dir_save_interm.glob('*.parquet')]
        if len(all_models) > 0:
            # compatibility with older version:
            if 'k_cluster' not in all_models[0].columns:
                del col_types['k_cluster']
                if verbose > 1:
                    print('Loading older version of models, `k_cluster` was not recorded')
            all_models = pd.concat(all_models, axis=0).astype(col_types)
            all_models.index = np.arange(len(all_models))
            return all_models
    
    # if no intermediate file was found or if recompute:
    if verbose > 0:
        print('searching hyperparameters')
    all_models = []
    if var_aggreg is None:
        print('To search for the best NAS models, please provide `var_aggreg`.')
        return
    if var_aggreg_samples_info is None:
        print('To search for the best NAS models, please provide `var_aggreg_samples_info`.')
        return
        
    # screen NAS parameters
    if iter_reducer_type is None:
        iter_reducer_type = ['umap']
    if iter_dim_clust is None:
        iter_dim_clust = [2, 3, 4, 5]
    if iter_n_neighbors is None:
        iter_n_neighbors = [15, 45, 75, 100, 200]
    if iter_k_cluster is None:
        iter_k_cluster = iter_n_neighbors
    if iter_metric is None:
        iter_metric = ['manhattan', 'euclidean', 'cosine']
    if iter_clusterer_type is None:
        iter_clusterer_type = ['hdbscan', 'spectral', 'ecg', 'leiden', 'gmm']
    if clust_size_params is None:
        clust_size_params = {
            'spectral': {
                'clust_size_param_name': 'n_clusters',
                'iter_clust_size_param': range(3, 20),
            },
            'leiden': {
                'clust_size_param_name': 'resolution',
                'iter_clust_size_param': [0.1, 0.03, 0.01, 0.003, 0.001],
            },
            'hdbscan': {
                'clust_size_param_name': 'min_cluster_size',
                'iter_clust_size_param': [50, 200],
            },
            'ecg': {
                'clust_size_param_name': 'ecg_ensemble_size',
                'iter_clust_size_param': [5, 10, 15, 20],
            },
            'gmm': {
                'clust_size_param_name': 'n_clusters',
                'iter_clust_size_param': range(3, 20),
            },
        }
    if iter_normalize is None:
        iter_normalize = ['total', 'niche', 'obs', 'clr', 'niche&obs']

    if show_progress:
        iter_reducer_type = tqdm(iter_reducer_type, leave=False)
    for reducer_type in iter_reducer_type:
        if reducer_type == 'none':
            iter_dim_clust_used = [0]
        else:
            iter_dim_clust_used = iter_dim_clust
        if show_progress:
            iter_dim_clust_used = tqdm(iter_dim_clust_used, leave=False)
        for dim_clust in iter_dim_clust_used:
            if show_progress:
                iter_n_neighbors = tqdm(iter_n_neighbors, leave=False)
            for n_neighbors in iter_n_neighbors:
                if show_progress:
                    iter_metric = tqdm(iter_metric, leave=False)
                for metric in iter_metric:
                    # avoid clustering given more neighbors than what was used for dim reduction
                    iter_k_cluster_used = [x for x in iter_k_cluster if x <= n_neighbors]
                    if show_progress:
                        iter_k_cluster_used = tqdm(iter_k_cluster_used, leave=False)
                    for k_cluster in iter_k_cluster_used:
                        if show_progress:
                            iter_clusterer_type = tqdm(iter_clusterer_type, leave=False)
                        for clusterer_type in iter_clusterer_type:
                            clust_size_param_name = clust_size_params[clusterer_type]['clust_size_param_name']
                            iter_clust_size_param = clust_size_params[clusterer_type]['iter_clust_size_param']

                            if show_progress:
                                iter_clust_size_param = tqdm(iter_clust_size_param, leave=False)
                            for clust_size_param in iter_clust_size_param:
                                cluster_params = {
                                    'reducer_type': reducer_type,
                                    'n_neighbors': n_neighbors, 
                                    'metric': metric,
                                    'min_dist': 0.0,
                                    'clusterer_type': clusterer_type, 
                                    'dim_clust': dim_clust, 
                                    'k_cluster': k_cluster, 
                                    # 'flavor': 'CellCharter',
                                    clust_size_param_name: clust_size_param,
                                }
                                str_params = '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
                                if verbose > 1:
                                    print(str_params)

                                cluster_labels, cluster_dir, nb_clust, _ = get_clusterer(var_aggreg, sof_dir, verbose=verbose, **cluster_params)
                                n_clusters = len(np.unique(cluster_labels))
                                
                                # Survival analysis (just heatmap for now)
                                niches = cluster_labels
                                if n_clusters >= min_cluster_pred:
                                    for normalize in iter_normalize:
                                        str_params = '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
                                        str_params = str_params + f'_normalize-{normalize}'

                                        results_path = dir_save_interm / f'{str_params}.parquet'
                                        new_model = None
                                        l1_ratio = np.nan
                                        alpha = np.nan
                                        score_roc_auc = np.nan
                                        score_ap = np.nan
                                        score_mcc = np.nan
                                        n_coefs = np.nan
                                        score_cic = np.nan

                                        if results_path.exists() and not recompute:
                                            if verbose > 2:
                                                print(f'load {results_path.stem}')
                                            new_model = pd.read_parquet(results_path)
                                        else:
                                            if train_model and n_clusters < max_cluster_pred:
                                                if verbose > 2:
                                                    print(f'compute {results_path.stem}')
                                        
                                                var_aggreg_niches = var_aggreg_samples_info.copy()
                                                var_aggreg_niches['niche'] = np.array(niches)

                                                counts = make_niches_composition(var_aggreg_niches[predict_key], niches, var_label=var_label, normalize=normalize)
                                                counts.index = counts.index.astype(status_pred.index.dtype)
                                                exo_vars = counts.columns.astype(str).tolist()

                                                df_surv = pd.concat([status_pred, counts], axis=1, join='inner').fillna(0)
                                                df_surv.columns = df_surv.columns.astype(str)
                                                df_surv.index.name = var_label
                                                if drop_nan:
                                                    n_obs_orig = len(df_surv)
                                                    df_surv.dropna(axis=0, inplace=True)
                                                    n_obs = len(df_surv)

                                                if pred_type == 'binary':
                                                    models = logistic_regression(
                                                        df_surv[exo_vars + [group_col]],
                                                        y_name=group_col,
                                                        col_drop=[var_label],
                                                        cv_train=cv_train, 
                                                        cv_adapt=cv_adapt, 
                                                        cv_max=cv_max,
                                                        plot_coefs=False,
                                                        split_train_test=split_train_test,
                                                        )
                                                    
                                                    score_roc_auc = np.nanmax([models[model_type]['score']['ROC AUC'] for model_type in models.keys()])
                                                    score_ap = np.nanmax([models[model_type]['score']['AP'] for model_type in models.keys()])
                                                    score_mcc = np.nanmax([models[model_type]['score']['MCC'] for model_type in models.keys()])
                                                    if verbose > 2:
                                                        print(f'score ROC AUCc: {score_roc_auc:.3f}')
                                                        print(f'score AP: {score_ap:.3f}')
                                                        print(f'score MCC: {score_mcc:.3f}')
                                                    
                                                    best_id = np.argmax([models[model_type]['score']['ROC AUC'] for model_type in models.keys()])
                                                    l1_ratio = [models[model_type]['model'].l1_ratio_[0] for model_type in models.keys()][best_id]
                                                    alpha = [models[model_type]['model'].C_[0] for model_type in models.keys()][best_id]
                                                    
                                                    if score_roc_auc >= min_score_plot:
                                                        if plot_heatmap:
                                                            # make folder to save figures
                                                            path_parts = cluster_dir.parts[-2:]
                                                            dir_save_figures = dir_save_interm
                                                            for part in path_parts:
                                                                dir_save_figures = dir_save_figures / part
                                                            dir_save_figures.mkdir(parents=True, exist_ok=True)

                                                            try:
                                                                g, d = plot_heatmap(
                                                                    df_surv[exo_vars + [group_col]].reset_index(), 
                                                                    obs_labels=var_label, 
                                                                    group_var=group_col, 
                                                                    groups=[0, 1],
                                                                    group_names=group_cat_mapper,
                                                                    figsize=(10, 10),
                                                                    z_score=False,
                                                                    cmap=sns.color_palette("Reds", as_cmap=True),
                                                                    return_data=True,
                                                                    )
                                                                figname = f"biclustering_{str_params}_roc_auc-{score_roc_auc:.3f}.jpg"
                                                                plt.savefig(dir_save_figures / figname, dpi=150)
                                                                plt.show()

                                                                g, d = plot_heatmap(
                                                                    df_surv[exo_vars + [group_col]].reset_index(), 
                                                                    obs_labels=var_label, 
                                                                    group_var=group_col, 
                                                                    groups=[0, 1],
                                                                    group_names=group_cat_mapper,
                                                                    figsize=(10, 10),
                                                                    z_score=1,
                                                                    cmap=sns.color_palette("Reds", as_cmap=True),
                                                                    return_data=True,
                                                                    )
                                                                figname = f"biclustering_{str_params}_roc_auc-{score_roc_auc:.3f}_col_zscored.jpg"
                                                                plt.savefig(dir_save_figures / figname, dpi=150)
                                                                plt.show()
                                                            except:
                                                                pass

                                                elif pred_type == 'survival':
                                                    # non_pred_cols = [patient_col, sample_col, event_col, duration_col]
                                                    # pred_cols = [x for x in df_surv if x not in non_pred_cols]

                                                    # Xt = OneHotEncoder().fit_transform(X)
                                                    Xt = df_surv[exo_vars]
                                                    Xt.columns = Xt.columns.astype(str)
                                                    y = surv_col_to_numpy(df_surv, event_col, duration_col)

                                                    # Search best CoxPH model
                                                    models = []
                                                    scores = []
                                                    all_cv_results = []
                                                    for l1_ratio in l1_ratios:
                                                        if verbose > 1:
                                                            print(f'l1_ratio: {l1_ratio}', end='; ')
                                                        
                                                        try:
                                                            coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_alpha, max_iter=100))
                                                            coxnet_pipe.fit(Xt, y)

                                                            # retrieve best alpha
                                                            estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                                                            # estimated_alphas = [0.1, 0.01]

                                                            cv = KFold(n_splits=5, shuffle=True, random_state=0)
                                                            gcv = GridSearchCV(
                                                                make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio)),
                                                                param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
                                                                cv=cv,
                                                                error_score=0.5,
                                                                n_jobs=n_jobs_gridsearch,
                                                            ).fit(Xt, y)

                                                            cv_results = pd.DataFrame(gcv.cv_results_)

                                                            # retrieve best model
                                                            best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]

                                                            models.append(best_model)
                                                            scores.append(best_model.score(Xt, y))
                                                            all_cv_results.append(cv_results)
                                                        except Exception as e:
                                                            print(e)
                                                    
                                                    if len(scores) > 0:
                                                        best_score_id = np.argmax(scores)
                                                        best_model = models[best_score_id]
                                                        best_cv = all_cv_results[best_score_id]
                                                        score_cic = scores[best_score_id] # concordance index right-censored
                                                        l1_ratio = best_model.l1_ratio
                                                        alpha = best_model.alphas[0]
                                                        best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])
                                                        non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
                                                        # print(f"Number of non-zero coefficients: {non_zero}")
                                                        non_zero_coefs = best_coefs.query("coefficient != 0")
                                                        coef_order = non_zero_coefs.abs().sort_values("coefficient").index
                                                        n_coefs = len(non_zero_coefs)

                                                        if plot_alphas:
                                                            alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
                                                            mean = cv_results.mean_test_score
                                                            std = cv_results.std_test_score

                                                            fig, ax = plt.subplots(figsize=(9, 6))
                                                            ax.plot(alphas, mean)
                                                            ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
                                                            ax.set_xscale("log")
                                                            ax.set_ylabel("concordance index")
                                                            ax.set_xlabel("alpha")
                                                            ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
                                                            ax.axhline(0.5, color="grey", linestyle="--")
                                                            ax.grid(True)

                                                        if plot_best_model_coefs:
                                                            _, ax = plt.subplots(figsize=(6, 8))
                                                            non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
                                                            ax.set_xlabel("coefficient")
                                                            ax.set_title(f'l1_ratio: {l1_ratio}  alpha: {alpha:.3g}  score: {score_cic:.3g}')
                                                            ax.grid(True)
                                            
                                            new_model = [dim_clust, n_neighbors, metric, clusterer_type, k_cluster, 
                                                         clust_size_param, n_clusters, normalize, l1_ratio, alpha]
                                            if pred_type == 'binary':
                                                new_model.extend([score_roc_auc, score_ap, score_mcc])
                                            elif pred_type == 'survival':
                                                new_model.extend([n_coefs, score_cic])
                                            new_model = pd.DataFrame(data=np.array(new_model).reshape((1, -1)), columns=columns)
                                            new_model = new_model.astype(col_types)
                                            new_model.to_parquet(results_path)
                                        
                                        all_models.append(new_model.values.ravel())

    all_models = pd.DataFrame(all_models, columns=columns)
    all_models = all_models.astype(col_types)
    all_models.to_parquet(aggregated_path)
    return all_models


def plot_screened_parameters(obj, cell_pos_cols, cell_type_col, orders, dim_clusts, processed_dir,
                             min_cluster_sizes, filter_samples=None, all_edges=None, sampling=False, var_type=None, 
                             n_neighbors=70, downsample=False, aggreg_dir=None, load_dir=None, save_dir=None, opt_str=''):
    """
    
    Example
    -------
    >>> processed_dir = Path('../data/processed/CODEX_CTCL')
    """

    # from skimage import color
    import colorcet as cc
  
    if var_type is None:
        var_type = 'markers'
    if aggreg_dir is None:
        aggreg_dir = processed_dir / "nas"
    if load_dir is None:
        load_dir = aggreg_dir / f"screening_dim_reduc_clustering_nas_on-{var_type}{opt_str}_n_neighbors-{n_neighbors}_downsample-{downsample}"
    if save_dir is None:
        save_dir = load_dir

    sample_ids = obj['Patients'].sort_values().unique()

    # For the visualization
    plots_marker = '.'
    size_points = 10
    if sampling is False:
        sampling = 1

    for order in orders:
        title = f"umap_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-2"
        file_path = str(save_dir / title) + '.csv'
        embed_viz = np.loadtxt(file_path, delimiter=',')

        n_cell_types = obj[cell_type_col].unique().size
        palette = sns.color_palette(cc.glasbey, n_colors=n_cell_types).as_hex()
        palette = [mpl.colors.rgb2hex(x) for x in mpl.cm.get_cmap('tab20').colors]

        for dim_clust in dim_clusts:
            print("dim_clust: {}".format(dim_clust))
            for min_cluster_size in min_cluster_sizes:
                print("    min_cluster_size: {}".format(min_cluster_size))

                # title = f"hdbscan_on-{var_type}_reducer-umap_nas{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                title = f"hdbscan_reducer-umap_nas_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                labels_hdbs = np.loadtxt(str(load_dir / title) + '.csv', delimiter=',')

                # Histogram of classes
                fig = plt.figure()
                class_id, class_count = np.unique(labels_hdbs, return_counts=True)
                plt.bar(class_id, class_count, width=0.8);
                plt.title('Clusters histogram');
                title = f"Clusters histogram - on {var_type} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
                plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
                plt.show(block=False)
                plt.close()   
                
                # make a cohort-wide cmap
                hdbs_cmap = cc.palette["glasbey_category10"]
                # make color mapper
                # series to sort by decreasing order
                uniq = pd.Series(labels_hdbs).value_counts().index.astype(int)
                n_colors = len(hdbs_cmap)
                labels_color_mapper = {x: hdbs_cmap[i % n_colors] for i, x in enumerate(uniq)}
                # make list of colors
                labels_colors = [labels_color_mapper[x] for x in labels_hdbs]
                labels_colors = pd.Series(labels_colors)

                for sample in sample_ids:
                    print("        sample: {}".format(sample))
                    select_sample = obj['Patients'] == sample
                    filenames = obj.loc[select_sample, 'FileName'].unique()

                    for filename in filenames:
                        if filter_samples is None or filename in filter_samples:
                            print("            filename: {}".format(filename))
                            select_file = obj['FileName'] == filename
                            select = np.logical_and(select_sample, select_file)

                            # load nodes and edges
                            if isinstance(all_edges, str):
                                file_path = processed_dir / all_edges / f'edges_sample-{filename}.csv'
                                pairs = pd.read_csv(file_path, dtype=int).values
                            else:
                                coords = obj.loc[select, cell_pos_cols].values
                                pairs = ty.build_delaunay(coords)
                                pairs = ty.link_solitaries(coords, pairs, method='knn', k=2)
                            # we drop z for the 2D representation
                            coords = obj.loc[select, ['x', 'y']].values

                            # Big summary plot
                            fig, ax = plt.subplots(1, 4, figsize=(int(7*4)+1, 7), tight_layout=False)
                            i = 0
                            ty.plot_network(coords, pairs, labels=obj.loc[select, 'ClusterName'], cmap_nodes=palette, marker=plots_marker, size_nodes=size_points, ax=ax[0])
                            ax[i].set_title('Spatial map of phenotypes', fontsize=14);

                            i += 1
                            ax[i].scatter(coords[:, 0], coords[:, 1], c=labels_colors[select], marker=plots_marker, s=size_points)
                            ax[i].set_title('Spatial map of detected areas', fontsize=14);
                            ax[i].set_aspect('equal')

                            i += 1
                            ax[i].scatter(embed_viz[select, 0], embed_viz[select, 1], c=labels_colors[select], s=5);
                            ax[i].set_title("HDBSCAN clustering on NAS", fontsize=14);
                            ax[i].set_aspect('equal')

                            i += 1
                            ax[i].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_colors);
                            ax[i].set_title("HDBSCAN clustering on NAS of all samples", fontsize=14);
                            ax[i].set_aspect('equal')
                            
                            # make plot limits equal
                            ax[i-1].set_xlim(ax[i].get_xlim())
                            ax[i-1].set_ylim(ax[i].get_ylim())

                            suptitle = f"Spatial omics data and detected areas - mean and std - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size} - sample {sample} - file {filename}";
                            fig.suptitle(suptitle, fontsize=18)

                            fig.savefig(save_dir / suptitle, bbox_inches='tight', facecolor='white', dpi=200)
                            plt.show(block=False)
                            plt.close()
