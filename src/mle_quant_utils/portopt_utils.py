import numpy as np
import pandas as pd
import datetime as dt

from tqdm import tqdm

def partial_dot_product(v, w):
    if isinstance(v, pd.DataFrame):
        assert v.shape[1] == 1, "v can only have 1 column"
        v = v.iloc[:, 0]

    if isinstance(w, pd.DataFrame):
        assert w.shape[1] == 1, "w can only have 1 column"
        w = w.iloc[:, 0]

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

from src.portfolio_opt import cvx_opt, cvx_opt_tc

def run_simple_backtesting(B_alpha, alpha_factor_name, risk_model, daily_returns, n_days_delay, *args, **kwargs):

    assert isinstance(daily_returns.index, pd.DatetimeIndex), "daily_returns must have a DatetimeIndex"

    STR_DATE_FMT = "%Y%m%d"
    port = {}
    bkt_dt_set = daily_returns.index[:-n_days_delay]
    bkt_dt_subset_loop = bkt_dt_set[:-n_days_delay]


    pnl = pd.DataFrame(index=bkt_dt_set, columns=['daily_pnl', 'daily_transaction_cost'])

    for bkt_dt in tqdm(bkt_dt_subset_loop, desc='Opt portfolio', unit='portfolio'):
        bkt_str_dt = dt.datetime.strftime(bkt_dt, STR_DATE_FMT)
        alpha_vector = B_alpha.loc[bkt_dt, alpha_factor_name].copy()
        opt = cvx_opt.OptimalHoldingsRegualization(**kwargs)
        w_opt = opt.find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'],
                         risk_model['idiosyncratic_var_vector'])
        port[bkt_str_dt] = w_opt

        # pnl
        idx_bkt_dt = bkt_dt_set.get_loc(bkt_dt)
        realization_dt = bkt_dt_set[idx_bkt_dt + n_days_delay]

        daily_rets = daily_returns.loc[realization_dt]
        port_pnl = partial_dot_product(v=w_opt, w=daily_rets)
        pnl.loc[bkt_dt, 'daily_pnl'] = port_pnl

    pnl['daily_transaction_cost'] = 0
    pnl['daily_total'] = pnl['daily_pnl'] - pnl['daily_transaction_cost']
    pnl['total'] = pnl['daily_total'].cumsum()

    return pnl, port


def run_backtesting(opt_engine, alpha_factor, risk_model, daily_returns, daily_adv, n_days_delay):

    assert isinstance(daily_returns.index, pd.DatetimeIndex), "daily_returns must have a DatetimeIndex"

    STR_DATE_FMT = "%Y%m%d"

    bkt_dt_set = daily_returns.index
    bkt_dt_subset_loop = bkt_dt_set[:-n_days_delay]
    port = {}
    w_prev = pd.Series(index=bkt_dt_set[0:1], data=[0.0])
    pnl_columns = ['returns_date', 'daily_pnl', 'daily_transaction_cost']
    pnl = pd.DataFrame(index=bkt_dt_subset_loop, columns=pnl_columns)

    for bkt_dt in tqdm(bkt_dt_subset_loop, desc='Opt portfolio', unit='portfolio'):
        bkt_str_dt = dt.datetime.strftime(bkt_dt, STR_DATE_FMT)
        alpha_vector = alpha_factor.loc[bkt_dt].copy()

        if hasattr(opt_engine, 'est_trans_cost'):
            adv_vector = daily_adv.loc[bkt_dt].copy()
            w_opt = opt_engine.find(alpha_vector, w_prev, adv_vector, risk_model['factor_betas'],
                             risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
            trans_cost = opt_engine.est_trans_cost(w_prev, w_opt, adv_vector)
        else:
            w_opt = opt_engine.find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'],
                         risk_model['idiosyncratic_var_vector'])
            trans_cost = 0.0

        port[bkt_str_dt] = w_opt

        # pnl
        idx_bkt_dt = bkt_dt_set.get_loc(bkt_dt)
        realization_dt = bkt_dt_set[idx_bkt_dt + n_days_delay]

        daily_rets = daily_returns.loc[realization_dt]
        port_pnl = partial_dot_product(v=w_opt, w=daily_rets)
        pnl.loc[bkt_dt, pnl_columns] = [realization_dt, port_pnl, trans_cost]

    pnl['daily_total'] = pnl['daily_pnl'] - pnl['daily_transaction_cost']
    pnl['total'] = pnl['daily_total'].cumsum()

    return pnl, port