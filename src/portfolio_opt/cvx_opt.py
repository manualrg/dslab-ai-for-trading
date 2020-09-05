import numpy as np
import pandas as pd
import cvxpy as cvx

from abc import ABC, abstractmethod


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
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

    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.Series(data=optimal_weights, index=alpha_vector.index)


class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
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

        return cvx.Minimize(-alpha_term)

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
    def _get_obj(self, weights, alpha_vector):
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
            assert(len(alpha_vector.columns) == 1), "alpha_vector should have only 1 column"
        else:
            assert isinstance(alpha_vector, pd.Series)

        alpha_term = alpha_vector.values.flatten() @ weights
        reg_term = self.lambda_reg * cvx.norm(weights, 2)
        return cvx.Minimize(- alpha_term + reg_term )

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
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
        assert (len(alpha_vector.columns) == 1)

        x_star = ((alpha_vector - alpha_vector.mean()) / alpha_vector.abs().sum()).values.reshape(-1)

        return cvx.Minimize(cvx.norm(weights - x_star, 2))