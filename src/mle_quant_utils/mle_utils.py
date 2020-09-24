import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error


# Region Prediction
def get_pred_alpha(preds, kind='clf'):
    """
    Rescale predicted probabilites into [-1+1]
    :param X_probas: 2d-array considering predicted probabilities
    :return: 1d array re-scaled to [-1+1]
    """
    if kind == 'clf':
        prob_array = [-1, 1]
        alpha_score = preds.dot(np.array(prob_array))
    elif kind == 'reg':
        alpha_score = preds
    else:
        print('Unknown kind: {}'.format(kind))
    return alpha_score


def neutralize_sector(factor_data, sector_data):
    """
    Neutralize by removing sector mean a factor
    :param factor_data: Factor returns
    :param sector_data: Sector encoding
    :return: factor returns sector de-meaned
    """
    for input_data in [factor_data, sector_data]:
        assert isinstance(input_data, pd.DataFrame), f"{input_data}  must be a pandas DataFrame"
        assert isinstance(input_data.index, pd.MultiIndex), f"{input_data} DataFrame index must be MultiIndex"
        assert isinstance(input_data.index.get_level_values(0), pd.DatetimeIndex),\
            f"{input_data} level=0 index must be DatetimeIndex"

    ml_alpha_sector = factor_data.join(sector_data, how='left')

    ml_alpha_sector_means = ml_alpha_sector.reset_index().set_index(['date', 'sector_code']).groupby(level=[0, 1])[
        ['p_test_champ']].transform(np.mean)
    ml_alpha_sector_means['asset'] = ml_alpha_sector.index.get_level_values(1)
    ml_alpha_sector_means = ml_alpha_sector_means.reset_index().set_index(['date', 'asset'])

    ml_alpha_neu = factor_data - ml_alpha_sector_means[['p_test_champ']]

    return ml_alpha_neu

def get_factor_alpha(preds_alpha, bins=9):
    assert isinstance(preds_alpha, pd.Series), "preds_alpha must be a pandas Series"
    assert isinstance(preds_alpha.index, pd.MultiIndex), "preds_alpha Series index must be MultiIndex"
    assert isinstance(preds_alpha.index.get_level_values(0), pd.DatetimeIndex),\
        "preds_alpha level=0 index must be DatetimeIndex"

    ranked = preds_alpha.groupby(level=0).transform(
        lambda grp: pd.cut(grp, bins=bins, labels=range(0, bins))
    )
    mu = ranked.groupby(level=0).transform(np.mean)
    sigma = ranked.groupby(level=0).transform(np.std)
    ml_alpha_test_zscored = (ranked - mu) / sigma
    return ml_alpha_test_zscored

def predict_and_score(model,  X_train, y_train, X_valid, y_valid, kind='clf'):
    results_cols = ['train_pmean', 'train_score', 'valid_pmean', 'valid_score', 'oob_score']

    p_train = pd.Series(index=X_train.index, data=model.predict(X_train))
    p_valid = pd.Series(index=X_valid.index, data=model.predict(X_valid))
    acc_train_idx_dict = {True: 'train_acc_target>0', False: 'train_acc_target<0'}
    acc_valid_idx_dict = {True: 'valid_acc_target>0', False: 'valid_acc_target<0'}
    if kind == 'clf':
        score_train = accuracy_score(y_train.values, p_train.values)
        acc_p_train = (p_train == y_train).groupby(y_train).mean().rename(index=acc_train_idx_dict)
        score_valid = accuracy_score(y_valid.values, p_valid.values)
        acc_p_valid = (p_valid == y_valid).groupby(y_valid).mean().rename(index=acc_valid_idx_dict)
    elif kind == 'reg':
        N_train = (y_train > 0).value_counts().rename(acc_train_idx_dict)
        score_train = mean_squared_error(y_train.values, p_train.values)
        acc_p_train = ((p_train > 0) == (y_train > 0)).groupby(y_train > 0).mean().rename(index=acc_train_idx_dict)
        acc_p_train['train_acc'] = (N_train*acc_p_train).sum()/N_train.sum()
        N_valid = (y_valid > 0).value_counts().rename(acc_valid_idx_dict)
        score_valid = mean_squared_error(y_valid.values, p_valid.values)
        acc_p_valid = ((p_valid > 0) == (y_valid > 0)).groupby(y_valid > 0).mean().rename(index=acc_valid_idx_dict)
        acc_p_train['valid_acc'] = (N_valid * acc_p_valid).sum() / N_valid.sum()
    else:
        print('Unknown kind: {}'.format(kind))

    if hasattr(model, "oob_score_"):
        result = pd.Series(index=results_cols,
                           data=[p_train.mean(), score_train, p_valid.mean(), score_valid, model.oob_score_])
    else:
        result = pd.Series(index=results_cols[:-1],
                           data=[p_train.mean(), score_train, p_valid.mean(), score_valid])
    result = result.append(acc_p_train).append(acc_p_valid)
    return result


# Region Sampling
def train_valid_test_split(all_x: pd.DataFrame, all_y: pd.DataFrame, train_size: float, valid_size: float,
                           test_size: float):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """

    assert isinstance(all_x.index, pd.MultiIndex)
    assert isinstance(all_x.index.get_level_values(0), pd.DatetimeIndex)
    assert isinstance(all_y.index, pd.MultiIndex)
    assert isinstance(all_y.index.get_level_values(0), pd.DatetimeIndex)
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0

    # Obtain an array of each split idx
    # Assume a MultiIndex pandas, level=0 is date
    dates_idx = all_x.index.get_level_values(0).unique()
    train_idx = int(train_size * len(dates_idx))
    valid_idx = train_idx + int(valid_size * len(dates_idx))

    # Split Multi index data
    train_dt_idx = dates_idx[:train_idx].values
    x_train = all_x.loc[train_dt_idx]
    y_train = all_y.loc[train_dt_idx]

    valid_dt_idx = dates_idx[train_idx: valid_idx].values
    x_valid = all_x.loc[valid_dt_idx]
    y_valid = all_y.loc[valid_dt_idx]

    test_dt_idx = dates_idx[valid_idx:].values
    x_test = all_x.loc[test_dt_idx]
    y_test = all_y.loc[test_dt_idx]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data

    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert isinstance(x.index, pd.MultiIndex)
    assert isinstance(x.index.get_level_values(0), pd.DatetimeIndex)
    assert isinstance(y.index, pd.MultiIndex)
    assert isinstance(y.index.get_level_values(0), pd.DatetimeIndex)
    assert len(x.shape) == 2
    assert len(y.shape) == 1

    # Get dates index
    # Assume that input dataframe is MultiIndex and level=0 is a date
    idx_dates = x.index.levels[0]
    n_dates = len(idx_dates)
    # Get a vector of indeces that represent the sample,
    # subset then from idx_date and cast to a list of pd.Timestamps
    subset_idx_dates = np.arange(start_i, n_dates, n_skip_samples + 1)
    subset_dates = [pd.Timestamp(dt) for dt in idx_dates[subset_idx_dates]]
    # Subset pandas DFs
    non_overlapping_x = x.loc[subset_dates]
    non_overlapping_y = y.loc[subset_dates]

    return non_overlapping_x, non_overlapping_y

# Region Assessment

def direction_accuracy_func(y_true, y_pred, **kwargs):
    kind = kwargs.get('kind', 'global')
    w_fp = kwargs.get('w_fp', 2)
    w_fn = kwargs.get('w_fn', 2)
    nobs = len(y_true)
    y_true_pos = (y_true > 0).astype(int)
    y_pred_pos = (y_pred > 0).astype(int)
    xtab = pd.crosstab(index=y_true_pos, columns=y_pred_pos)
    try:
        tp = xtab.loc[1, 1]
    except:
        tp = 0
    try:
        tn = xtab.loc[0, 0]
    except:
        tn = 0
    try:
        fp = xtab.loc[0, 1]
    except:
        fp = 0
    try:
        fn = xtab.loc[1, 0]
    except:
        fn = 0
    if kind == 'global':
        try:
            acc = (tp + tn) / nobs
        except:
            acc = 0
    elif kind == 'upwards':
        try:
            acc = tp / (tp + fn)
        except:
            acc = 0
    elif kind == 'downwards':
        try:
            acc = tn / (tn + fp)
        except:
            acc = 0
    elif kind == 'weighted':
        try:
            acc = (tp + tn) / (tp + tn + fp * w_fp + fn * w_fn)
        except:
            acc = 0
    else:
        raise ValueError("kind must be: global, upwards, downwards or weighted")

    return acc