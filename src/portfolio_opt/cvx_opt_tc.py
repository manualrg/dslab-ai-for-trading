import cvxpy as cvx
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector, w_prev, tc_lambda):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """

        raise NotImplementedError()

    @abstractmethod
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

        raise NotImplementedError()

    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T @ weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def _combi_series(self, x, y, fill_value=0.0):
        assert isinstance(x, pd.Series), 'x must be a pandas Series'
        assert isinstance(y, pd.Series), 'y must be a pandas Series'
        choose_left = lambda x, y: x
        w = x.combine(y, func=choose_left, fill_value=fill_value)
        w = w[y.index]

        return w

    def est_trans_cost(self, w_prev, w_opt, adv_vector):
        tc_lambda = self._compute_transaction_cost_constant(w_opt, adv_vector)
        est_trans_cost = self._compute_transaction_cost(w_opt, w_prev, w_opt, tc_lambda)
        est_zero_pos = abs(len(w_prev) - len(w_opt)) * tc_lambda.median()

        return est_zero_pos + est_trans_cost

    def _compute_transaction_cost_constant(self, alpha_vector, adv_vector):
        adv_vector_cp = self._combi_series(adv_vector, alpha_vector, fill_value=1.0e4)
        adv_vector_cp = np.clip(adv_vector_cp, a_min=1.0e4, a_max=np.inf)

        return 0.1 / adv_vector_cp

    def _compute_transaction_cost(self, weights, w_prev, alpha_vector, tc_lambda):
        w_pred_pad = self._combi_series(w_prev, alpha_vector, fill_value=0.0)
        w_delta_sq = (weights - w_pred_pad) ** 2
        trans_cost_term = w_delta_sq @ tc_lambda

        return trans_cost_term

    def find(self, alpha_vector, w_prev, adv_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        tc_lambda = self._compute_transaction_cost_constant(alpha_vector, adv_vector)
        obj = self._get_obj(weights, alpha_vector, w_prev, tc_lambda)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.Series(data=optimal_weights, index=alpha_vector.index)


class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector, w_prev, tc_lambda):
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
        trans_cost_term = self._compute_transaction_cost(weights, w_prev, alpha_vector, tc_lambda)

        return cvx.Minimize(-alpha_term + trans_cost_term)

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

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min


class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector, w_prev, tc_lambda):
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

        w_pred_pad = self._combi_series(w_prev, alpha_vector, fill_value=0.0)
        w_delta_sq = (weights - w_pred_pad) ** 2
        trans_cost_term = w_delta_sq @ tc_lambda

        return cvx.Minimize(- alpha_term + reg_term + trans_cost_term)

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55,
                 weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
