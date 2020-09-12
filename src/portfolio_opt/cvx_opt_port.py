import cvxpy as cvx
import numpy as np
import pandas as pd

class OptimalHoldingsRegualization():
    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0,
                 weights_max=0.55, weights_min=-0.55, min_adv=1e4):
        self.lambda_reg = lambda_reg
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        self.min_adv = min_adv

    def _get_obj(self, weights, alpha_vector, w_prev, adv_vector, transaction_cost):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : Single column DataFrame or Series
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        if isinstance(alpha_vector, pd.DataFrame):
            assert (len(alpha_vector.columns) == 1), "alpha_vector should have only 1 column"
        else:
            assert isinstance(alpha_vector, pd.Series)

        alpha_term = alpha_vector.values.flatten() @ weights
        reg_term = self.lambda_reg * cvx.norm(weights, 2)
        if transaction_cost:
            trans_cost_term = self._compute_transaction_cost(weights, w_prev, alpha_vector, adv_vector, self.min_adv)
        else:
            trans_cost_term = 0.

        return cvx.Minimize(- alpha_term + reg_term + trans_cost_term)

    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert (len(factor_betas.shape) == 2)

        constraints = [
            risk <= self.risk_cap ** 2,  # risk constraint
            factor_betas.T @ weights <= self.factor_max,  # factor exposures constraints
            factor_betas.T @ weights >= self.factor_min,
            weights <= self.weights_max,  # constraints on allocations
            weights >= self.weights_min,
            cvx.sum(cvx.abs(weights)) <= 1.0,  # leverage
            cvx.sum(weights) == 0.0,  # market neutral
        ]

        return constraints

    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T @ weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def _compute_transaction_cost_constant(self, alpha_vector, adv_vector, min_adv):
        """
        Get estimated adv vector with index aligned to alpha_vector, then obtain lambda term

        for asset[i] = 0.1 / est_adv(i,t); est_adv(i,t) = mean adv(i) in t, t-30
        Transaction cost model: 1% change in tradeSize(i,t) => 10 bps change in price(i,t)
                            tradeSize(i,t) = (h(i,t) - h(i,t-1)) / est_adv(i,t)
        :param alpha_vector: pandas Series that represent current set of assets
        :param adv_vector: pandas Series containing adv estimations
        :return: pandas Series with lambda constant for each asset
        """
        adv_vector_cp, _ = adv_vector.align(alpha_vector, join='right', fill_value=min_adv)
        adv_vector_cp = np.clip(adv_vector_cp, a_min=min_adv, a_max=np.inf)

        return 0.1 / adv_vector_cp

    def _compute_transaction_cost(self, weights, w_prev, alpha_vector, adv_vector, min_adv):
        tc_lambda = self._compute_transaction_cost_constant(alpha_vector, adv_vector, min_adv)
        w_prev_pad, _ = w_prev.align(alpha_vector, join='right', fill_value=0.0)

        w_delta_sq = (weights - w_prev_pad) ** 2
        trans_cost_term = w_delta_sq @ tc_lambda

        return trans_cost_term

    def find(self, alpha_vector, w_prev, adv_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector,
             transaction_cost=True, max_iters=500):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        obj = self._get_obj(weights, alpha_vector, w_prev, adv_vector, transaction_cost)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=max_iters)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.Series(data=optimal_weights, index=alpha_vector.index)

    def est_trans_cost(self, w_prev, w_opt, adv_vector):
        """
        Estimate transaction costs with a linear model. See: _compute_transaction_cost_constant
        :param w_prev: previous asset allocation h(i,t-1) measured in currency
        :param w_opt: current asset allocation h(i,t) measured in currency
        :param adv_vector: average dollar volume for each asset in portfolio, measured in the same currency as w_prev, w_opt
        :return: Estimated transactions costs in the same currency as w_prev, w_opt
        """
        # Compute trade: ||h(i,t) , h(i, t-1)||
        w_prev_pad, w_opt_pad = w_prev.align(w_opt, join='outer', fill_value=0.0)
        w_delta = (w_prev_pad-w_opt_pad)**2
        # Estimate lambda constant by asset
        adv_vector_cp, _ = adv_vector.align(w_delta, join='right', fill_value=self.min_adv)
        adv_vector_cp = np.clip(adv_vector_cp, a_min=self.min_adv, a_max=np.inf)
        tc_lambda = 0.1 / adv_vector_cp
        # Estimate transaction costs
        trans_cost_term = w_delta @ tc_lambda

        return trans_cost_term