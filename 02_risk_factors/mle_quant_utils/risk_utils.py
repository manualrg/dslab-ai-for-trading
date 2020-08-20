import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

# Region Risk

def compute_static_factor_returns(returns, factor_betas):
    """
    Given a set of fixed factor_betas (N, K) N-asset returns during a given period (from t=0 to T)
    estimate factor returns f(T, N) by running a linear regression for each time period to=0 to T,
    takin returns as dependent variable and factor_betas as independent variable
    :param returns: pandas DF, n_rows: trading days N, n_cols: Number of assets (N).
    :param factor_betas: pandas DF, n_rows: Number of assets (N), n_cols: Number of risk factors (K)
    :return: (fac_rets, exp_var)
        fac_rets: pandas DF. Estimated factor returns f(T, N)
        exp_var: pandas Series. Estimated explained variance by factor betas
    """
    assert returns.shape[1] == factor_betas.shape[0]

    dates_idx = returns.index
    fac_rets = pd.DataFrame(index=dates_idx, columns=factor_betas.columns)
    exp_var = pd.Series(index=dates_idx, name='exp_var')

    for dt in dates_idx:
        x_train = factor_betas  # factor betas may be dynamically updated everyday, this assumes them to be fixed
        y_train = returns.loc[dt]
        fac_ret_mod = LinearRegression(fit_intercept=False)
        fac_ret_mod.fit(x_train, y_train)
        fac_rets.loc[dt] = fac_ret_mod.coef_
        p_train = fac_ret_mod.predict(x_train)
        exp_var.loc[dt] = explained_variance_score(y_train, p_train)

    return fac_rets, exp_var

def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix
    F = Cov(f); variances at diagonal and covariances outside are zeroed
    Factor returns are assumed to be orthogonal when using a good risk model
    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns f(T,K)
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix F(K,K)
    """

    factor_cov_matrix = ann_factor * np.cov(factor_returns, ddof=1, rowvar=False)
    # Factor returns are assumed to be orthogonal when using PCA latent factor returns
    factor_cov_matrix = factor_cov_matrix * np.eye(factor_cov_matrix.shape[0])
    return factor_cov_matrix


def factor_var_vector(factor_returns, factor_cov_matrix):
    """
    Get the factor covariance diagonal as a vector

    Parameters
    ----------
    factor_returns : DataFrame
        daily factor returns, to fecth columns
    factor_cov_matrix : Ndarray 2-d
        Factor covariance matrix

    Returns
    -------
    factor_var_vector : DataFrame
        Factor Variances Vector (dim N)
    """

    factor_var_vector = np.diag(factor_cov_matrix)
    factor_var_vector = pd.DataFrame(data=factor_var_vector,
                                            index=factor_returns.columns)
    return factor_var_vector

def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix
    s(T,N): Specific returns -> S(N,N), variances at diagonal, zeros outside
    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix S(N,N)
    """

    # Implementation in slides: S(N,T)- B(NxK)·f(K,T)
    # Implementation considering dataframes as (T,N): f(T,K)·B.T(KxN)
    common_returns = factor_returns.dot(factor_betas.T)
    specific_returns = returns - common_returns

    s_var = ann_factor * np.var(specific_returns, axis=0, ddof=1).values
    idiosyncratic_var_matrix = np.diag(s_var)

    idiosyncratic_var_matrix = pd.DataFrame(
        data=idiosyncratic_var_matrix,
        index=returns.columns,
        columns=returns.columns
    )

    return idiosyncratic_var_matrix


def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector, fetch diagonal from S(N,N)

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector (dim N)
    """

    idiosyncratic_var_vector = np.diag(idiosyncratic_var_matrix)
    idiosyncratic_var_vector = pd.DataFrame(data=idiosyncratic_var_vector,
                                            index=returns.columns)
    return idiosyncratic_var_vector


def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Get the predicted portfolio risk

    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2

    bfb = (factor_betas.dot(factor_cov_matrix).dot(factor_betas.T) + idiosyncratic_var_matrix).values

    port_var = weights.values.T.dot(bfb).dot(weights.values).reshape(-1)[0]
    predicted_portfolio_risk = np.sqrt(port_var)

    return predicted_portfolio_risk

# Region PCA

def gridsearch_pca_expvar(X, n_components, k_folds=3, random_state = 123):
    exp_var = pd.DataFrame(index=n_components, columns=list(range(0, k_folds)))

    tscv = TimeSeriesSplit(k_folds)
    for k in n_components:
        pca = PCA(n_components=k, svd_solver='full', random_state=random_state)
        for idx_fold, (train_index, valid_index) in enumerate(tscv.split(X)):
            _X_train, _X_valid = X.iloc[train_index], X.iloc[valid_index]
            pca.fit(X.iloc[train_index])
            exp_var.loc[k, idx_fold] = pca.explained_variance_.sum()
    return exp_var

def fit_pca(returns, num_factor_exposures, svd_solver, random_state=123):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    pca = PCA(svd_solver=svd_solver, n_components=num_factor_exposures, random_state=random_state)
    pca.fit(returns)

    return pca


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas B(N,K) from the PCA model.
    pca.components_ (n_components: K, n_assets: N)

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame (n_rows: Number of assets-N, n_cols: Number of PCA components-K)
        Factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1

    factor_betas = pd.DataFrame(
        data=pca.components_.T,
        index=factor_beta_indices,
        columns=factor_beta_columns
    )
    return factor_betas


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.
    f(T,K)
    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame (n_rows: time points-T, n_cols: Number of PCA components-K)
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1

    factor_returns = pca.transform(returns)
    factor_returns = pd.DataFrame(
        data=factor_returns,
        index=factor_return_indices,
        columns=factor_return_columns
    )

    return factor_returns

