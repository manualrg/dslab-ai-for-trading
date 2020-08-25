import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

import graphviz
from IPython.display import Image
from sklearn.tree import export_graphviz

from tqdm import tqdm
import abc


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

# Region CV

def rf_train_val_grid_search(estimator, param_grid, X_train, y_train, X_valid, y_valid, kind='clf'):
    """
    Computes GridSearch on a RandomForsetClassifier-like estimator, given:
    :param estimator: RandomForsetClassifier-like
    :param param_grid: ParamGrid hyperparameter configurations
    :param X_train: array-like train features
    :param y_train: array-like train labels
    :param X_valid: array-like valid features
    :param y_valid: array-like valid labels
    :return: list of models and results pandas DF, joining hparams configurations and train/valid results
    """
    n_models = len(param_grid)
    results_cols = ['train_pmean', 'train_score', 'valid_pmean', 'valid_score', 'oob_score',
                    'train_acc_target>0', 'train_acc_target<0', 'valid_acc_target>0', 'valid_acc_target<0']
    if kind == 'reg':
        results_cols = results_cols + ['train_acc', 'valid_acc']

    results = pd.DataFrame(index=range(0, n_models), columns=results_cols)
    models, hparams_df_lst = [], []
    for i, hparams in enumerate(tqdm(param_grid, desc='Training Models', unit='Model')):
        rf_clf = clone(estimator).set_params(**hparams)
        res = rf_clf.fit(X_train, y_train)
        results.loc[i, :] = predict_and_score(res,  X_train, y_train, X_valid, y_valid, kind=kind)
        hparams_df_lst.append(pd.DataFrame(index=[i], data=hparams))
        models.append(res)
    return models, results.join(pd.concat(hparams_df_lst, axis=0), rsuffix="_hp")

def votrf_train_val_grid_search(param_grid, X_train, y_train, X_valid, y_valid, n_skip_samples=4, kind='clf'):
    """
    Computes GridSearch on a VotingClassifier estimator based on RandomForestClassifier, given:
    :param param_grid: ParamGrid hyperparameter configurations
    :param X_train: array-like train features
    :param y_train: array-like train labels
    :param X_valid: array-like valid features
    :param y_valid: array-like valid labels
    :param n_skip_samples: Each base estimator is computed on an independent sample from (X_train, y_train)
    :return: list of models and results pandas DF, joining hparams configurations and train/valid results
    """
    n_models = len(param_grid)
    results_cols = ['train_pmean', 'train_score', 'valid_pmean', 'valid_score', 'oob_score']
    results = pd.DataFrame(index=range(0, n_models), columns=results_cols)
    models, hparams_df_lst = [], []
    for i, hparams in enumerate(tqdm(param_grid, desc='Training Models', unit='Model')):
        vrf_clf = NoOverlapVoter(RandomForestClassifier(**hparams), n_skip_samples=n_skip_samples)
        res = vrf_clf.fit(X_train, y_train)
        results.loc[i, :] = predict_and_score(res,  X_train, y_train, X_valid, y_valid, kind=kind)
        hparams_df_lst.append(pd.DataFrame(index=[i], data=hparams))
        models.append(res)
    return models, results.join(pd.concat(hparams_df_lst, axis=0), rsuffix="_hp")

def vot_train_val_grid_search(estimator, param_grid, X_train, y_train, X_valid, y_valid, n_skip_samples=4):
    """
    Computes GridSearch on a VotingClassifier estimator based on RandomForestClassifier, given:
    :param param_grid: ParamGrid hyperparameter configurations
    :param X_train: array-like train features
    :param y_train: array-like train labels
    :param X_valid: array-like valid features
    :param y_valid: array-like valid labels
    :param n_skip_samples: Each base estimator is computed on an independent sample from (X_train, y_train)
    :return: list of models and results pandas DF, joining hparams configurations and train/valid results
    """
    n_models = len(param_grid)
    results_cols = ['train_pmean', 'train_score', 'valid_pmean', 'valid_score', 'oob_score']
    results = pd.DataFrame(index=range(0, n_models), columns=results_cols)
    models, hparams_df_lst = [], []
    for i, hparams in enumerate(tqdm(param_grid, desc='Training Models', unit='Model')):
        base_clf = clone(estimator).set_params(**hparams)
        vot_clf = NoOverlapVoter(base_clf, n_skip_samples=n_skip_samples)
        res = vot_clf.fit(X_train, y_train)
        results.loc[i, :] = predict_and_score(res,  X_train, y_train, X_valid, y_valid)
        hparams_df_lst.append(pd.DataFrame(index=[i], data=hparams))
        models.append(res)
    return models, results.join(pd.concat(hparams_df_lst, axis=0), rsuffix="_hp")

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
    n_rows = len(all_y)
    splits_idx = np.array([train_size, valid_size, test_size]) * n_rows
    splits_idx = np.array(splits_idx).cumsum().astype(np.int32)
    # extract date index and slice it according to previous indexes
    idx_dates = all_x.index.get_level_values(0)
    idx_train = idx_dates[:splits_idx[0]]
    idx_valid = idx_dates[splits_idx[0]:splits_idx[1]]
    idx_test = idx_dates[splits_idx[1]:]
    # Split Multi index data
    x_train = all_x.loc[idx_train[0]: idx_train[-1]]
    y_train = all_y.loc[idx_train[0]: idx_train[-1]]

    x_valid = all_x.loc[idx_valid[0]: idx_valid[-1]]
    y_valid = all_y.loc[idx_valid[0]: idx_valid[-1]]

    x_test = all_x.loc[idx_test[0]: idx_test[-1]]
    y_test = all_y.loc[idx_test[0]: idx_test[-1]]

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

# Region Bagging
def bagging_classifier(max_samples, max_features,parameters):
    """
    Build the bagging classifier.

    Parameters
    ----------
    n_estimators : int
        The number of base estimators in the ensemble
    max_samples : float
        The proportion of input samples drawn from when training each base estimator
    max_features : float
        The proportion of input sample features drawn from when training each base estimator
    parameters : dict
        Parameters to use in building the bagging classifier
        It should contain the following parameters:
            criterion
            min_samples_leaf
            oob_score
            n_jobs
            random_state

    Returns
    -------
    bagging_clf : Scikit-Learn BaggingClassifier
        The bagging classifier
    """

    required_parameters = {'criterion', 'min_samples_leaf', 'random_state',  # dtc hparams
                           'n_estimators',  'oob_score', 'n_jobs'  # rf hparams
    }
    assert not required_parameters - set(parameters.keys())

    dtc = DecisionTreeClassifier(criterion=parameters['criterion'],
                                 min_samples_leaf=parameters['min_samples_leaf'],
                                 random_state=parameters['random_state'])
    clf = BaggingClassifier(
        base_estimator=dtc,
        max_samples=max_samples,
        n_estimators=parameters['n_estimators'],
        max_features=max_features,
        random_state=parameters['random_state'],
        oob_score=parameters['oob_score'],
        n_jobs=parameters['n_jobs']
    )
    return clf

# Region VotingRF
class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError

    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError

    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf' + str(i), estimator) for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)

    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        if hasattr(self.estimators_[0], "oob_score_"):
            self.oob_score_ = self._calculate_oob_score(self.estimators_)

        return self

def calculate_oob_score(classifiers):
    """
    Calculate the mean out-of-bag score from the classifiers.

    Parameters
    ----------
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to calculate the mean out-of-bag score

    Returns
    -------
    oob_score : float
        The mean out-of-bag score
    """

    oob_scores = np.array([clf.oob_score_ for clf in classifiers])  # (n_clfs x n_oob_samples)
    oob_score = oob_scores.mean()
    return oob_score


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to fit on the non overlapping data
    n_skip_samples : int
        The number of samples to skip

    Returns
    -------
    fit_classifiers : list of Scikit-Learn Classifiers
        The classifiers fit to the the non overlapping data
    """

    # Generate N non-overlapping samples and fit a classifier for each
    n_clfs = len(classifiers)
    fit_classifiers = []
    for idx, offset in enumerate(range(0, n_clfs)):
        x_smpl, y_smpl = non_overlapping_samples(x, y, n_skip_samples, offset)
        clf = classifiers[idx]
        fit_classifiers.append(clf.fit(x_smpl, y_smpl))

    return fit_classifiers


class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)

# Region Visualization

def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))

