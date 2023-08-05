# -*- coding: utf-8 -*-
# @author: YanSen
# @date: 2023/05/03
from typing import List, Tuple

import numpy as np


class Problem(object):
    """
    Define the bounds of variables, objectives, and constraints using a vector that represents variables.

    The defined method is available for the linear objectives and constraints.
    If the problem is non-linear problem, the user need to overwrite get_individual_result() method
    to get the values of objectives and constraints with individual vector.

    """
    def __init__(
            self,
            n_vars,
            n_objs,
            n_constrs=0
    ):
        self._objs_info = []  # List to store objective function values
        self._constrs_info = []  # List to store constraint values
        self._vars_lb = - np.ones_like(n_vars) * np.inf  # Lower bound vector for variables
        self._vars_ub = np.ones_like(n_vars) * np.inf  # Upper bound vector for variables
        self._n_vars = n_vars
        self._n_objs = n_objs
        self._n_constrs = n_constrs

    def set_vars_bounds(self, lower_bounds: List[float], upper_bounds: List[float]) -> None:
        """
        Set the lower and upper bounds for each variable. If a variable has no lower or upper limit,
        the bound should be set to -np.inf or np.inf.

        Parameters:
            lower_bounds (List[float]): A list of float values representing the lower bounds for each variable.
            upper_bounds (List[float]): A list of float values representing the upper bounds for each variable.

        """
        self._vars_lb = np.array(lower_bounds)
        self._vars_ub = np.array(upper_bounds)

    def add_objectives(self, infos: List[Tuple[List[float], float]]) -> None:
        """
        Add objective functions to the problem.

        Parameters:
            infos (List[Tuple[List[float], float]]): List of tuples of coefficients and constant for the objective functions.

        """
        self._objs_info += infos

    def add_objective(self, coef: List[float], constant: float = 0) -> None:
        """
        Add an objective function to the problem.

        Parameters:
            coef (List[float]): List of coefficients for the linear objective function.
            constant (float): Constant term of the linear objective function.

        """
        self._objs_info.append((coef, constant))

    def add_constraints(self, infos: List[Tuple[List[float], float]]) -> None:
        """
        Add constraints to the problem in the format coef * individual + constant <= 0.

        Parameters:
            infos (List[Tuple[List[float], float]]): List of tuples of coefficients and constant for the linear constraints.

        """
        self._constrs_info += infos

    def add_constraint(self, coef: List[float], constant: float = 0) -> None:
        """
        Add a constraint to the problem in the format coef * individual + constant <= 0.

        Parameters:
            coef (List[float]): List of coefficients for the linear constraint.
            constant (float): Constant term of the linear constraint.

        """
        self._constrs_info.append((coef, constant))

    def get_individual_result(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate objective functions and violation of constraints for a given individual.

        Parameters:
            individual (np.ndarray): A numpy array representing the individual for which the problem is defined.

        """
        obj_result = np.array([np.dot(np.array(obj_coef), individual.T) + constant
                               for obj_coef, constant in self._objs_info])

        constr_result = np.array([np.dot(np.array(constr_coef), individual.T) + constant
                                  for constr_coef, constant in self._constrs_info])

        return obj_result, (constr_result > 0) * constr_result

    @property
    def n_vars(self) -> int:
        """
        Get the number of variables in the problem.

        """
        return self._n_vars

    @property
    def n_objs(self) -> int:
        """
        Get the number of objectives in the problem.

        """
        return self._n_objs

    @property
    def n_constrs(self) -> int:
        """
        Get the number of constraints in the problem.

        """
        return self._n_constrs

    @property
    def vars_lb(self) -> np.ndarray:
        """
        Get the lower bounds for each variable in the problem.

        """
        return self._vars_lb

    @property
    def vars_ub(self) -> np.ndarray:
        """
        Get the upper bounds for each variable in the problem.

        """
        return self._vars_ub
