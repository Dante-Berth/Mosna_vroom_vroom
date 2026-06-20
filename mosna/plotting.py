#! /usr/bin/env python3
"""mosna.plotting — visualisation helpers

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403

from mosna.clustering import make_cluster_cmap, make_niches_composition  # noqa: F401
from mosna.modeling import find_DE_markers, get_significant_coefficients  # noqa: F401
from mosna.utils import renormalize  # noqa: F401


def plot_niches_composition(counts=None, var=None, niches=None, var_label='variable', normalize='total', figsize=None):
    """
    Make a matrix plot of cell types composition of niches.
    """
    if counts is None:
        counts = make_niches_composition(var, niches, var_label='variable', normalize=normalize)
    
    plt.figure(figsize=figsize)
    fig = sns.heatmap(counts, linewidths=.5, cmap=sns.color_palette("Blues", as_cmap=True),
                      xticklabels=True, yticklabels=True)
    return fig


def plot_niches_histogram(niches, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    niche_id, niche_count = np.unique(niches, return_counts=True)
    ax.bar(niche_id, niche_count, width=0.8)
    ax.set_xticks(niche_id)
    return ax


def plot_pca(
    data: pd.DataFrame,
    pca: PCA_type = None,
    x_reduced: np.ndarray = None,
    n_components: int = 2,
    use_cols: Iterable = None,
    drop_cols: Iterable = None,
    show_var_names: bool = True,
    figsize: Tuple = (5, 5),
    scale_coords: int = True,
    group_var: str = None,
    groups: Iterable = None,
    group_colors: Union[str, Iterable] = None,
    groups_color_mapper: dict = None,
    groups_label_mapper: dict = None,
    legend: bool = True,
    legend_opt: dict = 'auto',
    show_grid: bool = True,
    ):
    """
    Perform Principal Components Analysis and plot observations in
    reduced dimensions and variables' contributions.

    Parameters
    ----------
    data : pd.DataFrame
        Data holding original variables.
    pca : PCA_type = None
        Pre-computed PCA model
    x_reduced : np.ndarray = None
        Pre-computed reduced coordinates.
    n_components : int = 2
        Number of PCA's components.
    use_cols : Iterable = None
        Variables to use for PCA.
    show_var_names : bool = True
        If True, display variables' names.
    figsize : Tuple = (7, 7)
        Figure size.
    scale_coords : int = True
        If True, coordinates are scaled with respect to the plot.
    group_var : str = None
        If provided, name of the column in `data` for groups
    groups : Iterable = None
        Observations' classes.
    group_colors : str = None
        If no group label is provided, a single color or an array of 
        colors, one for each observation.
    groups_color_mapper : dict = None
        Dictionnary mapping each class to a color.
    groups_label_mapper : dict = None
        Dictionnary mapping each class to a label.
    legend : bool = True
        If True, display a legend.
    legend_opt : dict, or str
        Position of the legend.
        If 'auto', sets the legend on the right of the axis.
    show_grid : bool = True
        If True, display a grid.
    """

    if group_var is not None:
        groups = data[group_var]
        # once we got groups, delete this columns from data for PCA
        if drop_cols is None:
            drop_cols = [group_var]
        else:
            drop_cols = drop_cols + [group_var]
    if groups_label_mapper is not None:
        groups = groups.map(groups_label_mapper)
    if use_cols is None:
        use_cols = data.columns
    if drop_cols is not None:
        use_cols = [col for col in use_cols if col not in drop_cols]
    if pca is None:
        sc = StandardScaler()
        X = data[use_cols]
        X = sc.fit_transform(X)
        pca = PCA(n_components=n_components)
    if x_reduced is None:
        x_reduced = pca.fit_transform(X)

    score = x_reduced[:, 0:2]
    coeff = np.transpose(pca.components_[0:2, :])

    # Get variance explained
    explained_var = pca.explained_variance_ratio_ * 100  # Convert to percentage

    xs = score[:, 0]
    ys = score[:, 1]
    n_var = coeff.shape[0]
    if scale_coords:
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
    else:
        scalex = 1.0
        scaley = 1.0

    if groups is not None:
        uniq_groups = np.unique(groups)
        nb_clust = len(uniq_groups)

        if groups_color_mapper is None:
            # choose colormap
            groups_cmap = mosna.make_cluster_cmap(uniq_groups)
            # make color mapper
            # series to sort by decreasing order
            n_colors = len(groups_cmap)
            groups_color_mapper = {x: groups_cmap[i % n_colors] for i, x in enumerate(uniq_groups)}
    else:
        if group_colors is None:
            group_colors = 'royalblue'
            
    fig, ax = plt.subplots(figsize=figsize)
    if groups is not None:
        for group_id in np.unique(groups):
            select = groups == group_id
            plt.scatter(score[select, 0]*scalex, score[select, 1]*scaley, 
                        c=groups_color_mapper[group_id], marker='.',
                        label=group_id);
        if legend:
            if legend_opt is None:
                plt.legend()
            else:
                if legend_opt == 'auto':
                    legend_opt = {'loc': 'center left', 'bbox_to_anchor': (1.05, 0.5)}
                plt.legend(**legend_opt)
    else:
        plt.scatter(score[:, 0]*scalex, embed_scoreviz[:, 1]*scaley, c=label_colors, marker='.');

    if show_var_names:
        for i in range(n_var):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
            if use_cols is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, use_cols[i], color = 'g', ha = 'center', va = 'center')
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    if show_grid:
        plt.grid()
    return fig, ax, pca, x_reduced


def plot_distrib_groups(
    data, 
    group_var, 
    groups=None,
    pval_data=None, 
    pval_col='pval_corr', 
    pval_thresh=0.05, 
    test='Mann-Whitney',
    max_cols=-1, 
    n_cols=None,
    exclude_vars=None, 
    id_vars=None, 
    var_name='variable', 
    value_name='value', 
    group_names=None,
    multi_ind_to_col=False, 
    scale_data=False,
    showfliers=False,
    figsize=(20, 6), 
    fontsize=20, 
    orientation=30, 
    palette='red_green', # or Set2
    legend_opt=None,
    ax=None,
    plot_type='boxplot', 
    add_points=True,
    ):
    """
    Plot the distribution of variables by groups.
    """

    data = data.copy()
    # Select variables that will be plotted
    if groups is None:
        groups = data[group_var].unique()
    if len(groups) == 2 and pval_data is not None:
        if isinstance(pval_data, str) and pval_data == 'compute':
            if 'other' in groups:
                # need to transform data
                data[group_var] = data[group_var].astype('category')
                select = data[group_var] != groups[1]
                data[group_var] = data[group_var].cat.add_categories("other")
                data.loc[select, group_var] = 'other'
                data[group_var] = data[group_var].cat.remove_unused_categories()
            pval_data = find_DE_markers(data, groups[0], groups[1], group_var=group_var, composed_order=0, test=test)
        nb_vars = np.sum(pval_data[pval_col] <= pval_thresh)
        print(f'There are {nb_vars} significant variables in `{pval_col}`')
        if n_cols is not None:
            nb_vars = n_cols
        else:
            if nb_vars == 0:
                nb_vars = len(pval_data)
            if max_cols > 0:
                nb_vars = min(nb_vars, max_cols)
        marker_vars = pval_data.sort_values(by=pval_col, ascending=True).head(nb_vars).index.tolist()
    else:
        marker_vars = data.columns.tolist()
        if max_cols > 0:
            marker_vars = marker_vars[:max_cols]
    # filter variable_names if exclude_vars was given
    if group_var in data.columns:
        gp_in_cols = [group_var] # exclude column of groups anyway
        if exclude_vars is None:
            exclude_vars = [group_var]
        else:
            exclude_vars = exclude_vars + [group_var]
    else:
        gp_in_cols = []
    if exclude_vars is not None:
        marker_vars = [x for x in marker_vars if x not in exclude_vars]
    
    # TODO: utility function to put id variables in multi-index into columns if not already in cols
    wide = data.loc[:, gp_in_cols + marker_vars]
    if id_vars is None:
        list_id_vars = list(wide.index.names) + gp_in_cols
        id_vars = [x for x in list_id_vars if x is not None]
    if multi_ind_to_col:
        wide = wide.reset_index()

    # select desired groups
    select = np.any([wide[group_var] == i for i in groups], axis=0)
    wide = wide.loc[select, :]

    if scale_data:
        wide.loc[:, marker_vars] = StandardScaler().fit_transform(wide.loc[:, marker_vars])

    long = pd.melt(
        wide, 
        id_vars=id_vars, 
        value_vars=marker_vars,
        var_name=var_name, 
        value_name=value_name)
    select = np.isfinite(long[value_name])
    long = long.loc[select, :]

    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax_none = False
    if len(groups) == 2:
        split = True
    else:
        split = False
    
    # manage colors
    if palette is not None:
        if isinstance(palette, str) and palette == 'red_green':
            palette = ['#F8766D', '#009E73']
            # else palette is the standard name of a palette
    
    # TODO: display variables on different axes if the have very differents ranges
    if plot_type == 'boxplot':
        sns.boxplot(x=var_name, y=value_name, hue=group_var, 
                    data=long, palette=palette, ax=ax, showfliers=showfliers);
    elif plot_type == 'violinplot':
        sns.violinplot(x=var_name, y=value_name, hue=group_var, 
                       data=long, palette=palette, split=split, ax=ax);
    if add_points:
        sns.stripplot(long, x=var_name, y=value_name, hue=group_var, 
                      dodge=True, size=4, palette='dark:.3', legend=None);
    plt.xticks(rotation=orientation, ha='right', fontsize=fontsize);
    plt.yticks(fontsize=fontsize);
    if group_names is not None:
        handles, previous_labels = ax.get_legend_handles_labels()
        try:
            new_labels = [group_names[x] for x in previous_labels]
        except KeyError:
            # keys and groups types are different, like str vs int
            to_type = type(previous_labels[0])
            # convert key types
            group_names = {to_type(key): val for key, val in group_names.items()}
            new_labels = [group_names[x] for x in previous_labels]
        if legend_opt is None:
            ax.legend(handles=handles, labels=new_labels)
        else:
            ax.legend(handles=handles, labels=new_labels, **legend_opt)
    if ax_none:
        return fig, ax


def plot_heatmap(
    data, 
    obs_labels=None, 
    group_var=None, 
    groups=None, 
    group_names=None,
    use_col=None, 
    skip_cols=[], 
    z_score=1, 
    drop_unique=True, 
    cmap=None, 
    center=None, 
    row_cluster=True, 
    col_cluster=True,
    palette='red_green', 
    figsize=(10, 10), 
    fontsize=10, 
    colors_ratio=0.03, 
    dendrogram_ratio=0.2, 
    cbar_kws=None,
    cbar_pos=(0.02, 0.8, 0.05, 0.18),
    legend_opt=None,
    legend_markersize=15,
    xlabels_rotation=30, 
    ax=None, 
    return_data=False,
    ):
    """
    Paameters
    ---------
    data : pd.DataFrame
        Table holding samples or patients clinical data and proportion
        of cells in niches.
    obs_labels : str, None
        Column of patient or sample IDs.
    group_var : str, None
        Column of clinical group (like responder vs non-responder)
    groups : Iterable, None
        Values of group to use for plotting, other values are ignored.
    group_names : dict, None
        Labels to display for each group.
    """

    data = data.copy(deep=True)
    # display(data.sample(3))
    if obs_labels is not None:
        data.index = data[obs_labels]
        data.drop(columns=[obs_labels], inplace=True)
    if use_col is None:
        skip_cols = skip_cols + [obs_labels, group_var]
        use_col = [x for x in data.columns if x not in skip_cols]
    else:
        data = data[use_col]
    if drop_unique:
        drop_cols = []
        keep_cols = []
        for col in use_col:
            n_uniq = len(data[col].unique())
            if n_uniq > 0:
                keep_cols.append(col)
            else:
                drop_cols.append(col)
        if len(drop_cols) > 0:
            print("Dropping colunms with unique value:")
            print(drop_cols)
            data = data.loc[:, keep_cols]
            use_col = keep_cols

    if group_var is not None:
        if groups is None:
            groups = data[group_var].unique()        
        # select desired groups
        data = data.query(f'{group_var} in @groups')
        # make lut group <--> color
        if palette is not None:
            if isinstance(palette, str) and palette == 'red_green':
                palette = ['#F8766D', '#009E73']
                # else palette is the standard name of a palette
            elif isinstance(palette, str) and palette == 'default':
                palette = sns.color_palette()
        lut = dict(zip(groups, palette))
        # Make the vector of colors
        colors = data[group_var].map(lut)
        data.drop(columns=[group_var], inplace=True)
    else:
        colors = None
    if cmap is None:
        if z_score is not None or (data.values.min() < 0 and data.values.max() > 0):
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            center = 0
        else:
            cmap = sns.light_palette("#C25539", as_cmap=True)
            center = None
    g = sns.clustermap(data, z_score=z_score, figsize=figsize, 
                       row_colors=colors, cmap=cmap, center=center,
                       row_cluster=row_cluster, col_cluster=col_cluster,
                       colors_ratio=colors_ratio, dendrogram_ratio=dendrogram_ratio,
                       cbar_pos=cbar_pos, cbar_kws=cbar_kws)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=fontsize);

    if colors is not None:
        all_markers = []
        all_labels = []
        for key, val in lut.items():
            all_markers.append(mpl.lines.Line2D([], [], marker="s", markersize=legend_markersize, linewidth=0, color=val))
            all_labels.append(key)
        if group_names is not None:
            all_labels = [group_names[x] for x in all_labels]
        if legend_opt is None:
            legend_opt = {'loc': 'lower right', 'bbox_to_anchor': (1.2, 1.1)}
        g.ax_heatmap.legend(all_markers, all_labels, **legend_opt)

    if hasattr(g, 'ax_row_colors') and colors is not None:
        g.ax_row_colors.set_xticklabels(g.ax_row_colors.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    if return_data:
        return g, data
    else:
        return g


def color_val_inf(val, thresh=0.05, col='green', col_back='white'):
    """
    Takes a scalar and returns a string with
    the css property `'color: green'` for values
    below a threshold, black otherwise.
    """
    color = col if val < thresh else col_back
    return 'color: %s' % color


def plot_survival_threshold(
    data: pd.DataFrame, 
    variable_name: str, 
    duration_col: str, 
    event_col: str, 
    thresh: float, 
    with_confidence: bool = True,
    colors: Union[str, list, None] = 'red_green',
    ax: plt.Axes = None,
    figsize: Iterable = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Kaplan-Meier curves of observations discriminated by a threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Survival data.
    variable_name : str
        Column to use in data. 
    duration_col : str
        Column for survival duration.
    event_col : str
        Column for survival event (death).
    thresh : float
        Threshold applied on variable.
    with_confidence : bool
        If True, KM curves are plotted with estimated confidence intervals.
    colors : list or None
        If not None, sets colors for patient groups.
    ax : plt.Axes
        Existing pyplot ax if any to draw KM curves.
    figsize: Iterable = (6, 4)
        Size of the figure to display.
    
    Returns
    -------
    fig : plt.Figure
        Figure of KM curves.
    ax : plt.Axes
        Axes of KM curves.
    """
    
    kmf_1 = KaplanMeierFitter()
    kmf_2 = KaplanMeierFitter()

    variable = data[variable_name]
    T = data[duration_col]
    E = data[event_col]
    select = (variable > thresh)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    kmf_1.fit(T[select], event_observed=E[select], label=f">   {thresh:.3g}")
    kmf_2.fit(T[~select], event_observed=E[~select], label=f"<= {thresh:.3g}")

    # modify default matplotlib colormaps to get correct colors
    if colors is not None:
        if isinstance(colors, str) and colors == 'red_green':
            color_inf = '#009E73' 
            color_sup = '#F8766D'
        else:
            color_inf = colors[0]
            color_sup = colors[1] 
    else:
            color_inf = None
            color_sup = None 
    # plot with correct cmap
    if with_confidence:
        kmf_1.plot_survival_function(ax=ax, color=color_sup)
        kmf_2.plot_survival_function(ax=ax, color=color_inf)
    else:
        kmf_1.survival_function_.plot(ax=ax, color=color_sup)
        kmf_2.survival_function_.plot(ax=ax, color=color_inf)

    ax.set_title(f"Survival given {variable_name}")
    return fig, ax


def plot_survival_coeffs(
    model, 
    data=None, 
    columns=None, 
    p_thresh=None,
    hazard_ratios=False, 
    sort_coefficients=True,
    colors=None, 
    min_size=1,
    max_size=5,
    auto_colors=False,
    grey_non_significant=True,
    default_color='royalblue',
    y_ticks_coeff=0.25,
    ax=None, 
    figsize=None,
    **errorbar_kwargs,
    ):
    """
    Produces a visual representation of the coefficients (i.e. log hazard ratios), including their standard errors and magnitudes.

    Parameters
    ----------
    model : lifeline object
        Trained lifeline CoxPH model.
    data : pd.DataFrame, None
        Survival data used to add more information on plots such as coeficients size.
    columns : list, optional
        specify a subset of the columns to plot
    p_thresh : float, None
        The p-value threshold used to filter out coefficients of the CoxPH model.
    hazard_ratios: bool, optional
        by default, ``plot`` will present the log-hazard ratios (the coefficients). However, by turning this flag to True, the hazard ratios are presented instead.
    sort_coefficients: bool, optional
        Sort coefficients for plotting.
    errorbar_kwargs:
        pass in additional plotting commands to matplotlib errorbar command

    Examples
    ---------
    >>> cph = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
    >>> cph.fit(df_surv, duration_col=duration_col, event_col=event_col, strata=strata)
    >>> ax = plot_survival_coeffs(cph, data=df_surv)

    Returns
    -------
    ax: matplotlib axis
        the matplotlib axis that be edited.

    """
    from matplotlib import pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    errorbar_kwargs.setdefault("c", "k")
    errorbar_kwargs.setdefault("fmt", "o")
    errorbar_kwargs.setdefault("markerfacecolor", "white")
    errorbar_kwargs.setdefault("markeredgewidth", 1.25)
    errorbar_kwargs.setdefault("elinewidth", 1.25)
    errorbar_kwargs.setdefault("capsize", None)

    z = inv_normal_cdf(1 - model.alpha / 2)

    if columns is None:
        columns = model.params_.index

    if p_thresh is not None:
        assert 0.0 < p_thresh < 1.0
        pval_columns = model.summary.index[model.summary['p'] <= p_thresh]
        columns = [x for x in columns if x in pval_columns]

    yaxis_locations = np.arange(len(columns))
    log_hazards = model.params_.loc[columns].values.copy()

    order = list(range(len(columns) - 1, -1, -1)) if not sort_coefficients else np.argsort(log_hazards)

    if colors is None:
        if auto_colors:
            cols_cmap = make_cluster_cmap(columns)
            # make color mapper
            # series to sort by decreasing order
            n_colors = len(cols_cmap)
            colors = [cols_cmap[i % n_colors] for i in range(len(columns))]
        else:
            colors = [default_color] * len(order)
    if grey_non_significant:
        coefs_sig = get_significant_coefficients(model.confidence_intervals_)
        # model[pval_col][i]<= 0.05
        colors = [x if columns[i] in coefs_sig else 'k' for i, x in enumerate(colors)]
    colors = np.array(colors)[order]
    if data is not None:
        weights = data.loc[:, columns].sum(axis=0)
        sizes = renormalize(np.array(weights), min_size, max_size)[order] * 50
        errorbar_kwargs['fmt'] = 'none'
    else:
        sizes = np.ones(len(columns)) * min_size * 50
    # errorbar_kwargs["s"] = sizes

    if hazard_ratios:
        exp_log_hazards = np.exp(log_hazards)
        upper_errors = exp_log_hazards * (np.exp(z * model.standard_errors_[columns].values) - 1)
        lower_errors = exp_log_hazards * (1 - np.exp(-z * model.standard_errors_[columns].values))
        ax.scatter(
            exp_log_hazards[order],
            yaxis_locations,
            c=colors, 
            s=sizes, 
            alpha=0.5,
            # cmap='viridis',
            )
        ax.errorbar(
            exp_log_hazards[order],
            yaxis_locations,
            xerr=np.vstack([lower_errors[order], upper_errors[order]]),
            **errorbar_kwargs,
        )
        ax.set_xlabel("HR (%g%% CI)" % ((1 - model.alpha) * 100))
    else:
        symmetric_errors = z * model.standard_errors_[columns].values
        ax.scatter(
            log_hazards[order],
            yaxis_locations,
            c=colors, 
            s=sizes, 
            alpha=0.5,
            # cmap='viridis',
            )
        ax.errorbar(
            log_hazards[order], 
            yaxis_locations, 
            xerr=symmetric_errors[order], 
            **errorbar_kwargs)
        ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - model.alpha) * 100))

    best_ylim = ax.get_ylim()
    ax.vlines(1 if hazard_ratios else 0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
    ax.set_ylim(best_ylim)

    tick_labels = [columns[i] for i in order]

    ax.set_yticks(yaxis_locations);
    ax.set_yticklabels(tick_labels);
    fig = ax.get_figure()
    fig.set_size_inches([6.4, y_ticks_coeff * len(columns)])

    return ax


def plot_clusters(embed_viz, 
                  cluster_labels=None, 
                  save_dir=None, 
                  cluster_params=None, 
                  extra_str='', 
                  show_id=True,
                  legend=True, 
                  legend_opt=None,
                  sort_legend=True,
                  cluster_colors=None,
                  aspect='equal',
                  return_cmap=False, 
                  figsize=(10,10),
                  ax=None,
                  ):
    """
    Plots clustered data on its 2D projection.

    Parameters
    ----------
    embed_viz : ndarray
        Mx2 array of points coordinates in the 2D projection.
    cluster_labels : array
        Cluster label ids of data points.
    save_dir : str or pathlib Path object
        Directory where the vizualisation is stored.
    cluster_params : dic
        Parameters used to generate the 2D projection and clustering to be
        included in the file name of saved figure.
    extra_str : str
        Additional string to add in the file name to indicate manual curation
        of clustering for instance.
    
    Returns
    -------
    fig, ax : matplotlib figure objects.
    """

    if cluster_labels is not None:
        nb_clust = cluster_labels.max()
        uniq_clusters = pd.Series(cluster_labels).value_counts().index
        if cluster_colors is None:
            # choose colormap
            clusters_cmap = make_cluster_cmap(uniq_clusters)
            # make color mapper
            # series to sort by decreasing order
            n_colors = len(clusters_cmap)
            cluster_colors = {x: clusters_cmap[i % n_colors] for i, x in enumerate(uniq_clusters)}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if cluster_labels is not None:
        for clust_id in uniq_clusters:
            select = cluster_labels == clust_id
            plt.scatter(embed_viz[select, 0], embed_viz[select, 1], 
                        c=cluster_colors[clust_id], marker='.',
                        label=clust_id);
        if legend:
            if legend_opt is None:
                legend_opt = {}
            plt.legend(**legend_opt)
            if sort_legend:
                # reorder legend labels
                handles, labels = ax.get_legend_handles_labels()
                labels = [int(x) for x in labels]
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, labels, **legend_opt)
    else:
        plt.scatter(embed_viz[:, 0], embed_viz[:, 1], c=cluster_colors, marker='.');
    plt.axis('off')
    if aspect == 'equal':
        ax.set_aspect('equal')

    if cluster_labels is not None and show_id:
        for clust_id in np.unique(cluster_labels):
            clust_targ = cluster_labels == clust_id
            x_mean = embed_viz[clust_targ, 0].mean()
            y_mean = embed_viz[clust_targ, 1].mean()
            plt.text(x_mean, y_mean, str(clust_id))

    if save_dir is not None:
        if cluster_params is None:
            str_params = ''
        else:
            str_params = '_' + '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
        figname =  f'cluster_labels{str_params}{extra_str}.png'
        plt.savefig(save_dir / figname, dpi=150)

    if return_cmap:
        return ax, cluster_colors
    return ax
