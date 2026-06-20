#! /usr/bin/env python3
"""mosna.modeling — survival, regression, classification and risk statistics

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403


def clean_data(
        data, 
        method='mixed', 
        thresh=1, 
        cat_cols=None, 
        modify_infs=True, 
        axis=0, 
        verbose=1):
    """
    Delete or impute missing or non finite data.
    During imputation, they are replaced by continuous values, not by binary values.
    We correct them into int values

    Parameters
    ----------
    data : dataframe
        Dataframe with nan or inf elements.
    method : str
        Imputation method or 'drop' to discard lines. 'mixed' allows
        to drop lines that have more than a given threshold of non finite values,
        then use the knn imputation method to replace the remaining non finite values.
    thresh : int or float
        Absolute or relative number of finite variables for a line to be conserved.
        If 1, all variables (100%) have to be finite.
    cat_cols : None or list
        If not None, list of categorical columns were imputed float values 
        are transformed to integers
    modify_infs : bool
        If True, inf values are also replaced by imputation, or discarded.
    axis : int
        Axis over which elements are dropped.
    verbose : int
        If 0 the function stays quiet.
    
    Return
    ------
    data : dataframe
        Cleaned-up data.
    select : If method == 'drop', returns also a boolean vector
        to apply on potential other objects like survival data.
    """
    if data.values.dtype != 'float':
        data = data.astype(float)
    to_nan = ~np.isfinite(data.values)
    nb_nan = to_nan.sum()
    if nb_nan != 0:
        if verbose > 0: 
            perc_nan = 100 * nb_nan / to_nan.size
            print(f"There are {nb_nan} non finite values ({perc_nan:.1f}%)")
        # set also inf values to nan
        if modify_infs:
            data[to_nan] = np.nan
        # convert proportion threshold into absolute number of variables threshold
        if method in ['drop', 'mixed'] and (0 < thresh <= 1):
            thresh = thresh * data.shape[axis]
        if method in ['drop', 'mixed']:
            # we use a custom code instead of pandas.dropna to return the boolean selector
            count_nans = to_nan.sum(axis=axis)
            select = count_nans <= thresh
            if axis == 0:
                data = data.loc[:, select].copy()
            else:
                data = data.loc[select, :].copy()
        # impute non finite values (nan, +/-inf)
        if method in ['knn', 'mixed']:
            if verbose > 0:
                print('Imputing data')
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            data.loc[:, :] = imputer.fit_transform(data.values)
            # set to intergers the imputed int-coded categorical variables
            # note: that's fine for 0/1 variables, avoid imputing on non binary categories
            if cat_cols is not None:
                data.loc[:, cat_cols] = data.loc[:, cat_cols].round().astype(int)
    if select is not None:
        return data, select
    else:
        return data


def make_clean_dummies(data, thresh=1, drop_first_binnary=True, verbose=1):
    """
    Delete missing or non finite categorical data and make dummy variables from them.
    Contrary to pandas' `get_dummy`, here nan values are not replaced by 0.

    Parameters
    ----------
    data : dataframe
        Dataframe of categorical data.
    thresh : int or float
        Absolute or relative number of finite variables for a line to be conserved.
        If 1, all variables (100%) have to be finite.
    drop_first_binnary : bool
        If True, the first dummy variable of a binary variable is dropped.
    verbose : int
        If 0 the function stays quiet.
    
    Return
    ------
    df_dum : dataframe
        Cleaned dummy variables.
    """

    # convert proportion threshold into absolute number of variables threshold
    if (0 < thresh <= 1):
        thresh = thresh * data.shape[1]
    # delete colums that have too many nan
    df_cat = data.dropna(axis=1, thresh=thresh)
    col_nan = df_cat.isna().sum()

    # one hot encoding of categories:
    # we make the nan dummy variable otherwise nan are converted and information is lost
    # then we manually change corresponding nan values and drop this column
    df_dum = pd.get_dummies(df_cat, drop_first=False, dummy_na=True)
    for col, nb_nan in col_nan.iteritems():
        col_nan = col + '_nan'
        if nb_nan > 0:
            columns = [x for x in df_dum.columns if x.startswith(col + '_')]
            df_dum.loc[df_dum[col_nan] == 1, columns] = np.nan
        df_dum.drop(columns=[col_nan], inplace=True)

    # Drop first class of binary variables for regression
    if drop_first_binnary:
        for col, nb_nan in col_nan.iteritems():
            columns = [x for x in df_dum.columns if x.startswith(col + '_')]
            if len(columns) == 2:
                if verbose > 0: 
                    print("dropping first class:", columns[0])
                df_dum.drop(columns=columns[0], inplace=True)
    return df_dum


def binarize_data(data, zero, one):
    """
    Tranform specific values of an array, dataframe or index into 0s and 1s.
    """
    binarized = deepcopy(data)
    binarized[data == zero] = 0
    binarized[data == one] = 1
    return binarized


def convert_quanti_to_categ(data, method='median'):
    """
    Transform continuous data into categorical data.
    """
    categ = {}
    if method == 'median':
        for col in data.columns:
            new_var = f'> med( {col} )'
            new_val = data[col] > np.median(data[col])
            categ[new_var] = new_val
    categ = pd.DataFrame(categ)
    return categ


def extract_X_y(data, y_name, y_values=None, col_names=None, col_exclude=None, binarize=True):
    """
    Extract data corresponding to specific values of a target variable.
    Useful to fit or train a statistical (learning) model. 

    Parameters
    ----------
    data : dataframe
        Data containing the X variables and target y variable
    y_name : str
        Name of the column or index of the target variable
    y_values : list or None
        List of accepted conditions to extract observations
    col_names : list or None
        List of variable to extract.
    col_exclude : list(str) or None
        Columns to ignore.
    binarize : bool
        If true and `y_values` has 2 elements, the vector `y` is
        binarized, with the 1st and 2nd elements of `y_values`
        tranformed into 0 and 1 respectivelly.
    
    Returns
    -------
    X : dataframe
        Data corresponding to specific target y values.
    y : array
        The y values related to X.
    """

    # if the y variable is in a pandas multiindex:
    if y_name not in data.columns and y_name in data.index.names:
        X = data.reset_index()
    else:
        X = deepcopy(data)
    if y_values is None:
        y_values = X[y_name].unique()
    if col_exclude is None:
        col_exclude = []
    col_exclude.append(y_name)
    if col_names is None:
        col_names = [x for x in data.columns if x not in col_exclude]

    # select desired groups
    select = np.any([X[y_name] == i for i in y_values], axis=0)
    y = X.loc[select, y_name]
    X = X.loc[select, col_names]
    if len(y_values) == 2 and binarize:
        y = binarize_data(y, zero=y_values[0], one=y_values[1])
    return X, y


def make_composed_variables(data, use_col=None, method='proportion', order=2):
    """
    Create derived or composed variables from simpler ones.  
    When producing ratios of variables, ratios of identical variables and
    reverse ratios are avoided, e.g. a/b, but no a/a nor b/a.
    When producing ratios of ratios of variables (order=2), equivalent and
    inverse ratios are avoided, e.g. (a/b)/(c/d), but no (a/b)/(a/d), and
    no (a/c)/(b/d).

    Example
    -------
    >>> df = pd.DataFrame({
            'a': [24, 24, 24],
            'b': [12, 8, 8],
            'c': [6, 4, 3],
            'd': [3, 4, 1],
        })
    >>> mosna.make_composed_variables(df)
       a / b  a / c  a / d     b / c  b / d  c / d  (a / b) / (c / d)
    0    2.0    4.0    8.0  2.000000    4.0    2.0                1.0
    1    3.0    6.0    6.0  2.000000    2.0    1.0                3.0
    2    3.0    8.0   24.0  2.666667    8.0    3.0                1.0                   
    """
    
    if use_col is None:
        use_col = data.columns
    if method == 'proportion':
        # ratio of variables
        new_vars = {}
        for i, var_1 in enumerate(use_col):
            for var_2 in use_col[i+1:]:
                new_var_name = f"{var_1} / ( {var_1} + {var_2} )"
                new_vars[new_var_name] = data[var_1] / (data[var_1] + data[var_2])
        new_data = pd.DataFrame(data=new_vars)
    elif method == 'ratio':
        # ratio of variables
        new_vars = {}
        for i, var_1 in enumerate(use_col):
            for var_2 in use_col[i+1:]:
                new_var_name = f"{var_1} / {var_2}"
                new_vars[new_var_name] = data[var_1] / data[var_2]
        new_data = pd.DataFrame(data=new_vars)
    
        if order == 2:
            # ratios of ratios of variables
            new_vars = {}
            for i, var_1 in enumerate(use_col):
                for j, var_2 in enumerate(use_col[i+1:]):
                    for k, var_3 in enumerate(use_col[i+j+2:]):
                        for var_4 in use_col[i+j+k+3:]:
                            pair_1 = [var_1, var_2]
                            pair_2 = [var_3, var_4]
                            new_var_name = f"({var_1} / {var_2}) / ({var_3} / {var_4})"
                            new_vars[new_var_name] = (data[var_1] / data[var_2]) / (data[var_3] / data[var_4])
            next_data = pd.DataFrame(data=new_vars)
            new_data = pd.concat([new_data, next_data], axis=1)


    return new_data


def find_DE_markers(
        data, 
        group_ref, 
        group_tgt, 
        group_var, 
        markers=None, 
        exclude_vars=None, 
        is_independent=True,
        patient_col=None,
        composed_vars=False, 
        composed_order=2, 
        test='Mann-Whitney', 
        fdr_method='indep', 
        alpha=0.05,
        ):
    

    if composed_vars:
        data = pd.concat([data, make_composed_variables(data, order=composed_order)], axis=1)
    if markers is None:
        markers = data.columns
    if isinstance(group_var, str):
        if group_var in data.columns:
            if exclude_vars is None:
                exclude_vars = [group_var]
            else:
                exclude_vars = exclude_vars + [group_var]
            group_var = data[group_var].values
        elif group_var in data.index.names:
            group_var = data.index.to_frame()[group_var]
        else:
            raise ValueError('The name of the group variable is not in columns nor in the index.')

    select_tgt = group_var == group_tgt
    if group_ref == 'other':
        select_ref = group_var != group_tgt
    elif not isinstance(group_ref, list):
        select_ref = group_var == group_ref
    else:
        select_ref = group_var == group_ref[0]
        for ref_id in group_ref[1:]:
            select_ref = np.logical_or(select_ref, group_var == ref_id)
    if isinstance(select_tgt, pd.Series):
        select_tgt = select_tgt.values
        select_ref = select_ref.values

    pvals = []
    # filter variable_names if exclude_vars was given
    if exclude_vars is not None:
        markers = [x for x in markers if x not in exclude_vars]
    used_markers = []
    if is_independent:
        for marker in markers:
            dist_tgt = data.loc[select_tgt, marker].dropna()
            dist_ref = data.loc[select_ref, marker].dropna()
            # select = np.logical_and(np.isfinite(dist_tgt), np.isfinite(dist_ref))
            # dist_tgt = dist_tgt[select]
            # dist_ref = dist_ref[select]
            if len(dist_tgt) > 0 and len(dist_ref) > 0:
                if test == 'Mann-Whitney':
                    mwu_stat, pval = mannwhitneyu(dist_tgt, dist_ref)
                if test == 'Welch':
                    w_stat, pval = ttest_ind(dist_tgt, dist_ref, equal_var=False)
                if test == 'Kolmogorov-Smirnov': 
                    ks_stat, pval = ks_2samp(dist_tgt, dist_ref)
                pvals.append(pval)
                used_markers.append(marker)
        pvals = pd.DataFrame(data=pvals, index=used_markers, columns=['pval'])
        pvals = pvals.sort_values(by='pval', ascending=True)
    else:
        y = group_var  # binary outcome
        X = data.loc[:, markers]
        # Add intercept to fixed effects
        X = add_constant(X)
        # exog_vc: variance components for random intercepts
        # One-hot encode patient IDs
        patients_dummies = pd.get_dummies(data.loc[:, patient_col], drop_first=False)
        # ident: all patient columns share the same variance parameter
        ident = np.zeros(patients_dummies.shape[1], dtype=int)
        # Fit Bayesian logistic mixed model
        model = BinomialBayesMixedGLM(y, X, patients_dummies, ident)
        result = model.fit_vb()  # variational Bayes for speed
        pvals = result.summary().tables[0]

    if fdr_method is not None and is_independent:
        rejected, pval_corr = fdrcorrection(pvals['pval'], method=fdr_method)
        pvals['pval_corr'] = pval_corr
    
    return pvals


def get_significant_coefficients(
    df: pd.DataFrame, 
    lower_col: str = '95% lower-bound', 
    upper_col: str = '95% upper-bound', 
    variables_in: str = 'index') -> Iterable[str]:
    """"
    Pick variables with significant coefficients in a predictive model.

    Parameters
    ----------
    df : dataframe
        Coefficients with lower and upper confidence intervals.
    lower_col : str
        Column storing the lower bound
    upper_col : str
        Column storing the upper bound
    variables_in : str
        If 'index', variables are selected from the index, otherwise
        they are selected from the column indicated by variables_in.
    
    Returns
    -------
    variables : Iterable[str]
        Variables with significant coefficients.
    """
    select = np.logical_or(
        np.logical_and(df[lower_col] < 0, df[upper_col] < 0),
        np.logical_and(df[lower_col] > 0, df[upper_col] > 0))
    if variables_in == 'index':
        variables = df.index.values[select]
    else:
        variables = df.loc[select, variables_in]
    return variables


def find_best_survival_threshold(
    data: pd.DataFrame, 
    variable_name: str, 
    duration_col: str, 
    event_col: str, 
    perc_range: Tuple[int, int, int] = (10, 91, 10)
    ) -> Tuple[float, int, float]:
    """
    Find the threshold that minimizes the p-value of the log-rank test.

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
    perc_range : Tuple[int, int, int]
        Parameters to generate the percentiles used as potential thresholds.
    
    Returns
    -------
    best_thresh : float
        Best threshold.
    best_perc : int
        Best percentile.
    best_pval : float
        Best p-value. 
    """
    
    variable = data[variable_name]
    T = data[duration_col]
    E = data[event_col]
    
    all_perc = np.arange(*perc_range)
    all_thresh = []
    all_pvals = []
    for perc in all_perc:
        thresh = np.percentile(variable, perc)
        select = (variable > thresh)
        p_val = logrank_test(T[select], T[~select], E[select], E[~select], alpha=.99).p_value
        all_thresh.append(thresh)
        all_pvals.append(p_val)

    best_idx = np.argmin(all_pvals)
    best_thresh = all_thresh[best_idx]
    best_perc = all_perc[best_idx]
    best_pval = all_pvals[best_idx]

    return best_thresh, best_perc, best_pval


def find_survival_variable(surv, X, reverse_response=False, return_table=True, return_model=True, model_kwargs=None, model_fit=None):
    """
    Fit a CoxPH model for each single variable, and detect the ones
    that are statistically significant.
    """
    pass


def forward_regression(X, y,
                       learner=sm.OLS, # sm.Logit
                       threshold_in=0.05,
                       verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            try:
                model = learner(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            except np.linalg.LinAlgError:
                print(f"LinAlgError with column {new_column}")
                new_pval[new_column] = 1
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included


def backward_regression(X, y,
                        learner=sm.OLS,
                        threshold_out=0.05,
                        verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = learner(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def stepwise_regression(X, y=None,
                        y_name=None,
                        y_values=None,
                        col_names=None,
                        learner=sm.OLS,
                        threshold_in=0.05,
                        threshold_out=0.05,
                        support=1,
                        verbose=False,
                        ignore_warnings=True,
                        kwargs_model={},
                        kwargs_fit={}):
    """
    Parameters
    ----------
    suport : int
        Minimal "support", i.e the minimal number of
        different values (avoid only 1s, etc...)
    """

    if y is None:
        X, y = extract_X_y(X, y_name, y_values)
    if col_names is not None:
        col_names = [x for x in X.columns if x in col_names]
        X = X[col_names]
    
    # drop variable that don't have enough different values
    if support > 0:
        drop_cols = []
        for col in X.columns:
            uniq = X[col].value_counts()
            if len(uniq) == 1:
                drop_cols.append(col)
            else:
                # drop variables with non-most numerous values are too few
                minority_total = uniq.sort_values(ascending=False).iloc[1:].sum()
                if minority_total < support:
                    drop_cols.append(col)
        if len(drop_cols) > 0:
            X.drop(columns=drop_cols, inplace=True)
            if verbose:
                print("Dropping variables with not enough support:\n", drop_cols)
        
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    initial_list = []
    included = list(initial_list)
    # record of dropped columns to avoid infinite cycle of adding/dropping
    drop_history = []
    
    while True:
        changed = False
        # ------ Forward selection ------
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            try:
                model = learner(y, sm.add_constant(pd.DataFrame(X[included+[new_column]])), **kwargs_model).fit(**kwargs_fit)
                new_pval[new_column] = model.pvalues[new_column]
            except np.linalg.LinAlgError:
                print(f"LinAlgError with column {new_column}")
                new_pval[new_column] = 1
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
            
            # ------ Backward selection ------
            while True:
                back_changed = False
                model = learner(y, sm.add_constant(pd.DataFrame(X[included])), **kwargs_model).fit(**kwargs_fit)
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max() # null if pvalues is empty
                if worst_pval > threshold_out:
                    worst_feature = pvalues.idxmax()
                    if worst_feature in drop_history:
                        changed = False # escape the forward/backward selection
                        if verbose:
                            print('Variable "{:30}" already dropped once, escaping adding/dropping cycle.'.format(worst_feature))
                    else: 
                        back_changed = True
                        included.remove(worst_feature)
                        drop_history.append(worst_feature)
                        if verbose:
                            print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                if not back_changed:
                    break
        
        if not changed:
            model = learner(y, sm.add_constant(pd.DataFrame(X[included])), **kwargs_model).fit(**kwargs_fit)
            return model, included


def logistic_regression(
    data, 
    y=None,
    y_name=None,
    y_values=None,
    col_drop=None, 
    cv_train=5, 
    cv_adapt=True, 
    cv_max=10, 
    l1_ratios_list='auto', 
    split_train_test=True,
    test_size=0.25,
    compare_null_model=True,
    patient_data=None,
    dir_save=None,
    plot_coefs=True,
    save_plot_coefs=False,
    save_coefs=False,
    save_scores=False,
    save_preds=False,
    plot_confusion_matrix=True,
    save_confusion_matrix=False,
    plot_ROC_curve=False,
    save_ROC_curve=True,
    display_nsamples=True,
    str_prefix='',
    figsize=(8, 8),
    verbose=1,
    ):
    """
    Train logistic regression models looking for the best hyperparameters
    for the ElasticNet penalization, and dislay or save results and models.

    Parameters
    ----------
    data : DataFrame
        Table containing predictive variables.
    y : array_like, optional
        Response / target variable if it is not included in `data`.
    y_name : str, optional
        If `y` is not provided, it is used to extract the response from `data`.
    y_values : list, optional
        List of accepted conditions to extract observations
        If provided, the fist value is set to zero and the second is set to one for prediction.
    col_drop : iterable, optional
        Columns to ignore in `data`

    Returns
    -------
    models : dict
        Record of scikit-learn's models and their associated performance and
        coefficients for each l1 ratios list.
    
    Example
    -------
    >>> col_drop = ['Patients', 'Spots']
    >>> y_name = 'Groups'
    """

    # Elasticnet logistic regression
    # l1_ratio = 0 the penalty is an L2 penalty (Ridge)
    # l1_ratio = 1 the penalty is an L1 penalty (Lasso)

    # Test either one of those combinations
    if l1_ratios_list == 'auto':
        l1_ratios_list = [
            ['default', [0.5]],
            ['naive', np.linspace(0, 1, 21)],           # naive param grid
            ['advised', [.1, .5, .7, .9, .95, .99, 1]], # advised in scikit-learn documentation
        ]

    score_labels = [
        'ROC AUC', # Receiver Operating Characteristic Area Under the Curve
        'AP',      # Average Precision
        'MCC',     # Matthews Correlation Coefficient
    ]

    # /!\ not related to `X = obj[aggreg_vars].values`
    if y is None:
        X, y = extract_X_y(data, y_name, y_values)

    X = data.reset_index()
    for col in col_drop:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    # # select groups
    X = X.drop(columns=[y_name])
    var_idx = X.columns

    start = time()

    models = dict()
    for l1_name, l1_ratios in l1_ratios_list:

        if split_train_test:
            # stratify train / test by response
            np.random.seed(0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=0, 
                shuffle=True, 
            )
        else:
            X_train = X
            X_test = X
            y_train = y
            y_test = y
        test_index = X_test.index
        # Standardize data to give same weight to regularization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if patient_data is not None:
            cv = GroupKFold(n_splits=cv_train)
            import sklearn
            sklearn.set_config(enable_metadata_routing=True)
            groups=patient_data

            np.random.seed(0)
            clf = linear_model.LogisticRegressionCV(
                cv=cv,
                Cs=20, 
                penalty='elasticnet', 
                # scoring='neg_log_loss', 
                scoring='roc_auc', 
                solver='saga', 
                l1_ratios=l1_ratios,
                max_iter=10000,
                n_jobs=-1,  # or n_jobs-1 to leave one core available
            )
            clf.fit(X, y, groups=groups)
            training_succeeded = not np.all(clf.coef_ == 0)
            sklearn.set_config(enable_metadata_routing=False)
        else:
            training_succeeded = False
            cv_used = cv_train
            while not training_succeeded and cv_used <= cv_max:
                np.random.seed(0)
                clf = linear_model.LogisticRegressionCV(
                    cv=cv_used,
                    Cs=20, 
                    penalty='elasticnet', 
                    # scoring='neg_log_loss', 
                    scoring='roc_auc', 
                    solver='saga', 
                    l1_ratios=l1_ratios,
                    max_iter=10000,
                    n_jobs=-1,  # or n_jobs-1 to leave one core available
                )
                clf = clf.fit(X_train, y_train)
                training_succeeded = not np.all(clf.coef_ == 0)
                if not training_succeeded:
                    if cv_adapt:
                        cv_used += 1
                        print(f"        training failed, trying with cv = {cv_used}")
                    else:
                        print(f"        training failed")
                        break
        
        models[l1_name] = {'model': clf}

        if training_succeeded:
            if patient_data is not None:
                # Predictions using group-aware CV
                y_pred_proba = y.copy()
                y_pred_proba.iloc[:] = 0
                y_pred = y.copy()
                y_pred.iloc[:] = 0
                for train_idx, test_idx in cv.split(X, y, groups):
                    sklearn.set_config(enable_metadata_routing=True)
                    clf_fold = linear_model.LogisticRegressionCV(
                        cv=cv,
                        Cs=[clf.C_[0]],
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=l1_ratios,
                        scoring="roc_auc",
                        max_iter=10000,
                        n_jobs=-1
                    )

                    clf_fold.fit(X.loc[train_idx, :], y[train_idx], groups=groups[train_idx])
                    y_pred_proba.loc[test_idx] = clf_fold.predict_proba(X.loc[test_idx, :])[:, 1]
                    y_pred.loc[test_idx] = clf.predict(X.loc[test_idx, :])
                    sklearn.set_config(enable_metadata_routing=False)

            else:
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                y_pred = clf.predict(X_test)

            score = {
                'ROC AUC': metrics.roc_auc_score(y_test, y_pred_proba),
                'AP' : metrics.average_precision_score(y_test, y_pred_proba),
                'MCC': metrics.matthews_corrcoef(y_test, y_pred),
            }
            if score['ROC AUC'] <= 0.5:
                compare_null_model = False

            # Save model coefficients and plots
            l1_ratio = np.round(clf.l1_ratio_[0], decimals=4)
            C = np.round(clf.C_[0], decimals=4)

            coef = pd.DataFrame({'coef': clf.coef_.flatten()}, index=var_idx)
            coef['abs coef'] = coef['coef'].abs()
            coef = coef.sort_values(by='abs coef', ascending=False)
            coef['% total'] = coef['abs coef'] / coef['abs coef'].sum()
            coef['cum % total'] = coef['% total'].cumsum()
            coef['coef OR'] = np.exp(coef['coef'])
            nb_coef = coef.shape[0]
            if save_coefs:
                coef.to_csv(dir_save / f"LogisticRegressionCV_coefficients.csv")
        
            fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred_proba)
            j_scores = tpr - fpr  # Youden's J = sensitivity - (1 - specificity)
            best_roc_threshold = roc_thresholds[np.argmax(j_scores)]
            roc_auc = metrics.auc(fpr, tpr)
            y_pred = (y_pred_proba >= best_roc_threshold).astype(int)

            preds = pd.DataFrame(data={'y_pred_proba': y_pred_proba,
                                        'y_pred': y_pred,
                                        'y_test': np.array(y_test)},
                                    index=test_index)
            models[l1_name]['preds'] = preds
            models[l1_name]['l1_ratio'] = l1_ratio
            models[l1_name]['C'] = C
            models[l1_name]['best_roc_threshold'] = best_roc_threshold
            if save_preds:
                preds.to_csv(dir_save / f'{str_prefix}logistic_regression_predictions_{l1_name}.csv')
            
            if plot_coefs:
                nb_coef_plot = min(20, nb_coef)
                labels = coef.index[:nb_coef_plot]

                fig, ax = plt.subplots(figsize=(nb_coef_plot, 6))
                ax = coef.loc[labels, 'coef'].to_frame().plot.bar(ax=ax, color='#a6a6a6')
                ax.hlines(y=0, xmin=0, xmax=nb_coef_plot-1, colors='gray', linestyles='dashed')
                ticks_pos = np.linspace(start=0, stop=nb_coef_plot-1, num=nb_coef_plot)
                # ticks_label = np.round(ticks_label, decimals=2)
                ax.set_xticks(ticks_pos);
                # ax.set_xticklabels(ticks_label)
                ax.set_xticklabels(labels, rotation=45, ha='right');
                ax.set_xlabel('variables')
                ax.set_ylabel('coef')
                ax.set_title(f" l1_ratio {l1_ratio}, C {C}, AUC {score['ROC AUC']:.3f}")
                if save_plot_coefs:
                    fig.savefig(
                        dir_save / f"{str_prefix}logistic_regression_coefficients_grid-{l1_name}.jpg", 
                        bbox_inches='tight', 
                        facecolor='white', 
                        dpi=150,
                        )

            if plot_ROC_curve:
                fig_roc, ax_roc = plt.subplots(figsize=figsize)
                if display_nsamples:
                    add_str = f"\n(n_test_samples={len(y_test)})"
                else:
                    add_str = ''
                ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.3f}){add_str}')
                ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax_roc.legend(loc='best')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f"ROC curve for {l1_name}")
                if save_ROC_curve:
                    fig_roc.savefig(
                        dir_save / f"{str_prefix}logistic_regression_ROC_curve_grid-{l1_name}.jpg", 
                        bbox_inches='tight', 
                        facecolor='white', 
                        dpi=150,
                        )
            
            if plot_confusion_matrix:
                cm = confusion_matrix(y_test, y_pred)

                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Pred 0', 'Pred 1'],
                            yticklabels=['True 0', 'True 1'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix (Threshold = {best_roc_threshold:.2f})')
                if save_confusion_matrix:
                    fig_cm.savefig(
                        dir_save / f"{str_prefix}logistic_regression_confusion_matrix_grid-{l1_name}.jpg", 
                        bbox_inches='tight', 
                        facecolor='white', 
                        dpi=150,
                        )
        else:
            score = {
                'ROC AUC': np.nan,
                'AP' : np.nan,
                'MCC': np.nan,
            }
            coef = None
            print(f"        training failed with cv <= {cv_max}")

        if compare_null_model:
            y_shuffled = shuffle(y, random_state=0)

            if split_train_test:
                # stratify train / test by response
                np.random.seed(0)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_shuffled, 
                    test_size=test_size, 
                    random_state=0, 
                    shuffle=True, 
                )
            else:
                X_train = X
                X_test = X
                y_train = y_shuffled
                y_test = y_shuffled
            test_index = X_test.index
            # Standardize data to give same weight to regularization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)   

            clf.fit(X_train, y_train)

            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred_proba)
            j_scores = tpr - fpr  # Youden's J = sensitivity - (1 - specificity)
            best_roc_threshold = roc_thresholds[np.argmax(j_scores)]
            roc_auc = metrics.auc(fpr, tpr)
            y_pred = (y_pred_proba >= best_roc_threshold).astype(int)

            score['ROC AUC Null Model'] = roc_auc
            score['AP Null Model'] = metrics.average_precision_score(y_test, y_pred_proba)
            score['MCC Null Model'] = metrics.matthews_corrcoef(y_test, y_pred)
                
        models[l1_name]['score'] = score
        models[l1_name]['coef'] = coef

        if save_scores:
            scores = pd.DataFrame.from_dict(score, orient='index')
            scores.to_csv(dir_save / f'{str_prefix}logistic_regression_scores_grid-{l1_name}.csv')

    end = time()
    duration = end - start
    if verbose > 0:
        print(f"Training took {duration:.3f}s")

    return models


def train_XGBoost(
    data: pd.DataFrame, 
    y: Iterable = None,
    ) -> xgboost.core.Booster:
    """
    Train an XGBoost model.

    Parameters
    ----------
    data : DataFrame
        Table containing predictive variables.
    y : array_like, optional
        Response / target variable.

    Returns
    -------
    model : xgboost
        Trained XGBoost model
    """
    # training code here
    pass


def relative_risk(expo, nonexpo, alpha_risk=0.05):
    """
    Compute the relative risk between exposed and non exposed conditions.
    Diseases is coded as True or 1, healthy is codes as False or 0.
    alpha is the risk, default is 0.05.
    
    Example
    -------
    >>> expo = np.array([1, 1, 0, 0])
    >>> nonexpo = np.array([1, 0, 0, 0])
    >>> relative_risk(expo, nonexpo)
    """
        
    # number of exposed
    Ne = expo.size
    # number of diseased exposed
    De = expo.sum()
    # number of healthy exposed
    He = Ne - De
    # number of non-exposed
    Nn = nonexpo.size
    # number of diseased non-exposed
    Dn = nonexpo.sum()
    # number of healthy non-exposed
    Hn = Nn - Dn
    # relative risk
    RR = (De / Ne) / (Dn / Nn)
    
    # confidence interval
    eff = np.sqrt( He / (De * Ne) + Hn / (Dn + Nn))
    Z_alpha = np.array(stats.norm.interval(1 - alpha_risk, loc=0, scale=1))
    interv = np.exp( np.log(RR) + Z_alpha * eff)
    
    return RR, interv


def make_expo(control, test, filter_obs=None):
    """
    Make arrays of exposed and non-exposed samples from a control
    array defining the exposure (True or 1) and a test array defining
    disease status (True or 1).
    """
    
    # filter missing values
    select = np.logical_and(np.isfinite(control), np.isfinite(test))
    # combine with given selector
    if filter_obs is not None:
        select = np.logical_and(select, filter_obs)
    control = control[select]
    test = test[select]
    control = control.astype(bool)
    test = test.astype(bool)
    
    expo = test[control]
    nonexpo = test[~control]
    return expo, nonexpo


def make_risk_ratio_matrix(data, y_name=None, y_values=None, rows=None, columns=None, 
                           alpha_risk=0.5, col_filters={}):
    """
    Make the matrices of risk ratio and lower and upper bounds
    of confidence intervals.
    
    col_filters is a dictionnary to select observations for a given set of columns.
    the keys are the conditionnal columns, values are dictionaries which keys are either
    'all' to apply selector to all target columns of several taret columns names.
    """
    if y_name is not None:
        X, y = extract_X_y(data, y_name, y_values, binarize=False)
        X[y_name] = y
        data = X
    if rows is None:
        rows = data.columns
    if columns is None:
        columns = data.columns
    N = len(columns)
    rr = pd.DataFrame(data=np.zeros((N, N)), index=columns, columns=columns)
    rr_low = rr.copy()
    rr_high = rr.copy()

    for i in rows:
        for j in columns:
            # i tells what variable is used to define exposure
            # j is used to define disease status
            if i == j:
                rr.loc[i, j], (rr_low.loc[i, j], rr_high.loc[i, j]) = 1, (1, 1)
            else:
                if i in col_filters:
                    if 'all' in col_filters[i]:
                        filter_obs = col_filters[i]['all']
                        expo, nonexpo = make_expo(data[i], data[j], filter_obs=filter_obs)
                    elif j in col_filters[i]:
                        filter_obs = col_filters[i][j]
                        expo, nonexpo = make_expo(data[i], data[j], filter_obs=filter_obs)
                    else:
                        expo, nonexpo = make_expo(data[i], data[j])
                else:
                    expo, nonexpo = make_expo(data[i], data[j])
                rr.loc[i, j], (rr_low.loc[i, j], rr_high.loc[i, j]) = relative_risk(expo, nonexpo, alpha_risk=alpha_risk)
    # significance matrix
    rr_sig = (rr_low > 1) | (rr_high < 1)
    
    return rr, rr_low, rr_high, rr_sig
