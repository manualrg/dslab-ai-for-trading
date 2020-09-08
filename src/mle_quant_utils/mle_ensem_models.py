import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

import abc
from tqdm import tqdm

from src.mle_quant_utils import mle_utils

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

# Region Voting Ensembles

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
        x_smpl, y_smpl = mle_utils.non_overlapping_samples(x, y, n_skip_samples, offset)
        clf = classifiers[idx]
        fit_classifiers.append(clf.fit(x_smpl, y_smpl))

    return fit_classifiers


class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)

class NoOverlapVoterRegressorAbstract(VotingRegressor):
    @abc.abstractmethod
    def _calculate_oob_score(self, regressors):
        raise NotImplementedError

    @abc.abstractmethod
    def _non_overlapping_regressors(self, x, y, regressors, n_skip_samples):
        raise NotImplementedError

    def __init__(self, base_estimator, n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('reg' + str(i), base_estimator) for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        self.base_estimator = base_estimator
        super().__init__(estimators)

    def fit(self, X, y, sample_weight=None):
        estimator_names, regs = zip(*self.estimators)

        clone_regs = [clone(reg) for reg in regs]
        self.estimators_ = self._non_overlapping_regressors(X, y, clone_regs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        if hasattr(self.estimators_[0], "oob_score_"):
            self.oob_score_ = self._calculate_oob_score(self.estimators_)

        return self

class NoOverlapVoterRegressor(NoOverlapVoterRegressorAbstract):
    def _calculate_oob_score(self, regressors):
        return calculate_oob_score(regressors)

    def _non_overlapping_regressors(self, x, y, regressors, n_skip_samples):
        return non_overlapping_estimators(x, y, regressors, n_skip_samples)

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
        results.loc[i, :] = mle_utils.predict_and_score(res,  X_train, y_train, X_valid, y_valid, kind=kind)
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
        results.loc[i, :] = mle_utils.predict_and_score(res,  X_train, y_train, X_valid, y_valid, kind=kind)
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
        results.loc[i, :] = mle_utils.predict_and_score(res,  X_train, y_train, X_valid, y_valid)
        hparams_df_lst.append(pd.DataFrame(index=[i], data=hparams))
        models.append(res)
    return models, res