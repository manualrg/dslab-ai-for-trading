import pytest
import pandas as pd
import numpy as np
from src.portfolio_opt import cvx_opt_port

assets_lst = ['A', 'B', 'C']
n_assets = len(assets_lst)
n_beta_factors = 2
n_periods = 2
start_dt = '2020-09-07'

@pytest.fixture()
def get_returns_data():
    dates = pd.date_range(start_dt, periods=n_periods, freq='B')
    returns_data = np.random.normal(0, 1, (n_periods, n_assets))
    bkt_daily_returns = pd.DataFrame(index=dates, data=returns_data, columns=assets_lst)

    return bkt_daily_returns

@pytest.fixture()
def get_adv_vector():
    adv_data = pd.Series(index=assets_lst, data=np.random.normal(1e5, 1e2, n_assets), name='adv')
    return adv_data

@pytest.fixture()
def get_alpha_vector():
    return pd.Series(index=assets_lst, data=range(1, n_assets+1), name='alpha')

@pytest.fixture()
def get_risk_model():
    betas_data = np.random.normal(0, 1, (n_assets, n_beta_factors))
    returns_data = np.random.normal(0, 1, (n_periods, n_assets))

    idiosyncratic_var_vector = pd.DataFrame(index=assets_lst, data=abs(np.random.normal(0, 1, n_assets)))
    factor_var_vector = pd.DataFrame(index=range(0, n_beta_factors), data=range(1, n_beta_factors+1)[::-1])
    factor_betas = pd.DataFrame(index=assets_lst, data=betas_data, columns=range(0, n_beta_factors))
    factor_cov_matrix = np.cov(returns_data) * np.identity(returns_data.shape[0])

    return {
        'idiosyncratic_var_vector': idiosyncratic_var_vector,
        'factor_var_vector': factor_var_vector,
        'factor_betas': factor_betas,
        'factor_cov_matrix': factor_cov_matrix
    }


def get_weights(assets):
    n = len(assets)
    w_raw = np.random.uniform(0, 1, n)
    w = w_raw / np.sum(w_raw)

    return pd.Series(index=assets, data=w)

@pytest.mark.parametrize("in_wprev_assets", [['A', 'B', 'C'], ['A', 'B', 'C', 'X'], ['A', 'B']])
def test_w_prev_and_current_conformance(get_returns_data, get_adv_vector, get_alpha_vector, get_risk_model,
                                        in_wprev_assets):

    adv_vector = get_adv_vector
    adv_vector.name = 'adv_vector'
    alpha_vector = get_alpha_vector
    alpha_vector.name = 'alpha_vector'
    risk_model = get_risk_model
    w_prev = get_weights(in_wprev_assets)
    w_prev.name = 'w_prev'

    opt_engine_tc = cvx_opt_port.OptimalHoldingsRegualization(lambda_reg=0.0)
    w_opt = opt_engine_tc.find(alpha_vector, w_prev, adv_vector, risk_model['factor_betas'],
                            risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
    w_opt.name = 'w_opt'
    tc = opt_engine_tc.est_trans_cost(w_prev, w_opt, adv_vector)

    assert len(w_opt) == n_assets, f"w_opt:{w_opt}, n_assets: {n_assets}"
    assert tc > 0., f"transactions cost: {tc}"





