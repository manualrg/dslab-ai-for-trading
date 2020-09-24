import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm, t, gaussian_kde
import statsmodels.api as sm

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

def sharpe_ratio(returns, ann_period):
    """
    Compute (annualized) sharpe ratio (risk adjusted returns) on a series of returns
    VaR: statistic measuring maximum portfolio loss at a particular confidence level
    :param returns: pandas Series with pct returns
    :param ann_period: input to annualization factor, 12 if monthly, 252 if daily, so on
    :return: sharpe ratio (float)
    """
    return np.sqrt(ann_period) * returns.mean() / returns.std()

def compute_empirical_var(losses, ci):
    """
    Compute empirical VaR (Value at Risk) at a given confidence level
    :param losses:
    :param ci:
    :return:
    """
    return np.quantile(losses, ci)


def compute_theoretical_var(losses, ci, distribution='normal'):
    """
    Compute theoretical VaR (Value at Risk) at a given confidence level
    :param losses:
    :param ci:
    :return:
    """
    allowed_distributions = ['normal', 'student', 'gaussian_kde']
    assert distribution in allowed_distributions, f'distribution should be in : {allowed_distributions}'

    if distribution == allowed_distributions[0]:
        pm, ps = losses.mean(), losses.std()
        res = norm.ppf(ci, loc=pm, scale=ps)
    if distribution == allowed_distributions[1]:
        fitted = t.fit(losses)
        res = t.ppf(ci, *fitted)
    if distribution == allowed_distributions[2]:
        fitted = gaussian_kde(losses)
        sample = fitted.resample(100000)
        res = np.quantile(sample, ci)

    return res


def compute_cvar(losses, ci, distribution='normal'):
    """
    CVaR: measures expected loss given a minimum loss equal to the (theoretical) VaR
    :param losses:
    :param ci:
    :return:
    """
    allowed_distributions = ['normal', 'student']
    assert distribution in allowed_distributions, f'distribution should be in : {allowed_distributions}'

    if distribution == allowed_distributions[0]:
        pm, ps = losses.mean(), losses.std()
        var = norm.ppf(ci, loc=pm, scale=ps)
        tail_loss = norm.expect(lambda x: x, loc=pm, scale=ps, lb=var)
    if distribution == allowed_distributions[1]:
        fitted = t.fit(losses)
        var = t.ppf(ci, *fitted)
        tail_loss = t.expect(lambda y: y, args=(fitted[0],), loc=fitted[1], scale=fitted[2], lb=var)
    cvar = (1 / (1 - ci)) * tail_loss

    return cvar

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


def run_backtesting(opt_engine, flg_trans_cost, alpha_factor, risk_model, daily_returns, daily_adv, n_days_delay):

    assert isinstance(daily_returns.index, pd.DatetimeIndex), "daily_returns must have a DatetimeIndex"

    STR_DATE_FMT = "%Y%m%d"

    bkt_dt_set = daily_returns.index
    bkt_dt_subset_loop = bkt_dt_set[:-n_days_delay]
    port = {}
    w_prev = pd.Series(index=bkt_dt_set[0:1], data=[0.0])
    pnl_columns = ['returns_date', 'daily_pnl', 'daily_transaction_cost']
    pnl = pd.DataFrame(index=bkt_dt_subset_loop, columns=pnl_columns, dtype=float)

    for bkt_dt in tqdm(bkt_dt_subset_loop, desc='Opt portfolio', unit='portfolio'):
        bkt_str_dt = dt.datetime.strftime(bkt_dt, STR_DATE_FMT)
        alpha_vector = alpha_factor.loc[bkt_dt].copy()

        adv_vector = daily_adv.loc[bkt_dt].copy()
        w_opt = opt_engine.find(alpha_vector, w_prev, adv_vector, risk_model['factor_betas'],
                                risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'], flg_trans_cost)

        if flg_trans_cost:
            adv_vector = daily_adv.loc[bkt_dt].copy()
            trans_cost = opt_engine.est_trans_cost(w_prev, w_opt, adv_vector)
        else:
            trans_cost = 0.0

        port[bkt_str_dt] = w_opt

        # pnl
        idx_bkt_dt = bkt_dt_set.get_loc(bkt_dt)
        realization_dt = bkt_dt_set[idx_bkt_dt + n_days_delay]

        daily_rets = daily_returns.loc[realization_dt]
        port_pnl = partial_dot_product(v=w_opt, w=daily_rets)
        pnl.loc[bkt_dt, pnl_columns] = [realization_dt, port_pnl, trans_cost]

    pnl['returns_date'] = pd.to_datetime(pnl['returns_date'])
    pnl['daily_total'] = pnl['daily_pnl'] - pnl['daily_transaction_cost']
    pnl['accum_total'] = pnl['daily_total'].cumsum()
    pnl['accum_transaction_cost'] = pnl['daily_transaction_cost'].cumsum()

    return pnl, port

def join_weights_and_pnl(w_opt_df, pnl, returns):
    daily_returns_stack = returns.tz_localize(tz=None).stack()
    daily_returns_stack.name = 'returns'
    daily_returns_stack.index.names = ['date', 'asset']

    pnl_select = pnl[['returns_date', 'daily_total', 'accum_total']].tz_localize(tz=None)
    pnl_select['returns_date'] = pnl_select['returns_date'].dt.tz_localize(tz=None)

    pnl_and_w_df = w_opt_df.reset_index().merge(pnl_select, how='left', left_on='date', right_index=True). \
        merge(daily_returns_stack, how='left', left_on=['returns_date', 'asset'], right_index=True).\
        set_index(['date', 'asset'])

    pnl_and_w_df['daily_asset_pnl'] = pnl_and_w_df['w_opt'] * pnl_and_w_df['returns']
    pnl_and_w_df['position'] = pnl_and_w_df['w_opt'].apply(lambda x: '1.long' if x > 0 else '2.short')
    pnl_and_w_df['abs_w_opt'] = abs(pnl_and_w_df['w_opt'])

    return pnl_and_w_df


def portfolio_analyzer(weights: dict, pnl: pd.DataFrame, returns: pd.DataFrame, factor_betas: pd.DataFrame, factor_alphas: pd.DataFrame,
                       n_days_delay: float):
    assert isinstance(weights, dict), f"{weights} must be a dictionary of pandas Series"

    w_daily = {}
    all_factors = {}
    factors_returns = {}
    factor_exp = {}
    factor_returns_models = {}
    S_idiosyncratic_returns = pd.Series(dtype=float, name='idiosyncratic_returns')

    dates_lst = list(weights.keys())
    for i, key_str_dt in enumerate(dates_lst[:-n_days_delay]):

        key_dt = dt.datetime.strptime(key_str_dt, "%Y%m%d")
        key_tzaw_rets_dt = pd.Timestamp(dates_lst[i+n_days_delay]).tz_localize(tz="utc")
        # weights
        val_w = weights[key_str_dt]
        w_daily[key_dt] = val_w
        # factors
        B_alpha = factor_alphas.loc[key_dt].add_prefix('alpha_')
        B_beta = factor_betas.add_prefix('beta_')
        B_all_factors = B_alpha.join(B_beta)  # static betas
        all_factors[key_dt] = B_all_factors
        # returns
        returns_day = returns.loc[key_tzaw_rets_dt]
        returns_day.name = 'returns'
        returns_day.index.name = 'asset'

        # Compute factor returns f(t)[i]: r(t+n)[i] = b(t)[i]*f(t)[i] + s(t)[i] as reg coefs
        exog = B_all_factors.filter(regex='(^alpha|^beta)').fillna(0.0)
        endog = returns_day[exog.index].fillna(0.0)
        model = sm.OLS(endog, exog)
        res = model.fit()
        factor_returns_models[key_dt] = res
        factors_returns[key_dt] = res.params
        # Compute factor exposure e(t)[i] = w(t)[i]*f(t)[i]
        factor_exp[key_dt] = get_factor_exposures(factor_betas=exog, weights=val_w)
        # Compute idiosyncratic returns:  s(t)[i]*w(t)[i]
        S_idiosyncratic_returns[key_dt] = partial_dot_product(v=res.resid, w=val_w)

    w_opt_df = pd.concat(w_daily)
    w_opt_df.index.names = ['date', 'asset']
    w_opt_df.name = 'w_opt'

    B_factors_asset_df = pd.concat(all_factors)
    B_factors_asset_df.index.names = ['date', 'asset']

    f_factors_returns_df = pd.concat(factors_returns).unstack()
    f_factors_returns_df.index.name = 'date'
    E_factors_exp_df = pd.concat(factor_exp).unstack()
    E_factors_exp_df.index.name = 'date'

    S_idiosyncratic_returns.index.name = 'date'

    pnl_and_w_df = join_weights_and_pnl(w_opt_df, pnl, returns)

    return w_opt_df, pnl_and_w_df, B_factors_asset_df, f_factors_returns_df, E_factors_exp_df, S_idiosyncratic_returns


def compute_pnl_factor_attribution(E_factor_exposure, f_factor_returns):
    return (E_factor_exposure * f_factor_returns).cumsum()


def portfolio_attribution(E_factor_exposure, f_factor_returns, S_idiosyncratic_returns,
                          alpha_prefix='alpha', beta_prefix='beta'):

    alpha_attr = compute_pnl_factor_attribution(E_factor_exposure.filter(regex=fr'^{alpha_prefix}'),
                                                f_factor_returns.filter(regex=fr'^{alpha_prefix}'))

    beta_attr = compute_pnl_factor_attribution(E_factor_exposure.filter(regex=fr'^{beta_prefix}'),
                                                f_factor_returns.filter(regex=fr'^{beta_prefix}'))

    alpha_attr = alpha_attr.sum(axis=1)
    alpha_attr.name = 'alpha_pnl'

    beta_attr = beta_attr.sum(axis=1)
    beta_attr.name = 'beta_pnl'

    idiosyncratic_attr = S_idiosyncratic_returns.cumsum()
    idiosyncratic_attr.name = 'idiosyncratic_pnl'

    port_attr = pd.concat([alpha_attr, beta_attr, idiosyncratic_attr], axis=1)
    return port_attr
