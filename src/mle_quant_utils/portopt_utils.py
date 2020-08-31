import numpy as np


def partial_dot_product(v, w):
    common = v.index.intersection(w.index)
    return np.sum(v[common] * w[common])

def get_factor_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)

def get_portfolio_alpha_exposure(B_alpha, h_star):
    """
    Calculate portfolio's Alpha Exposure

    Parameters
    ----------
    B_alpha : patsy.design_info.DesignMatrix
        Matrix of Alpha Factors

    h_star: Numpy ndarray
        optimized holdings

    Returns
    -------
    alpha_exposures : Pandas Series
        Alpha Exposures
    """
    if isinstance(B_alpha, pd.Series):
        alpha_factor_names = [B_alpha.name]
    elif isinstance(B_alpha, pd.DataFrame):
        alpha_factor_names = B_alpha.columns
    else:
        alpha_factor_names = range(0, B_alpha.shape[1])
    return pd.Series(np.matmul(B_alpha.transpose(), h_star), index=alpha_factor_names)