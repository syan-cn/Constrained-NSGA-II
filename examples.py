# -*- coding: utf-8 -*-
# @author: YanSen
# @date: 2023/05/03

from typing import Tuple
import numpy as np
from problem import Problem
from nsga2 import ConstrainedNSGA2


def example_problem_1():
    """
    Define example problem1 1 with 2 decision variables, 2 objectives and 2 constraints.

    """

    problem1 = Problem(n_vars=2, n_objs=2, n_constrs=2)
    problem1.set_vars_bounds([0, 0], [1, 1])

    problem1.add_objective([1, 2])
    problem1.add_objective([-1, -2])

    problem1.add_constraint([1, -1], 0.5)
    problem1.add_constraint([-1, 2], 1)

    return problem1


def ex2_get_individual_result(individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the objective values and constraint violations for the given individual in example problem 2.

    Parameters:
        individual: The individual to be evaluated.

    Returns:
        tuple: A tuple containing the objective values and constraint violations.

    """
    g = np.sum([(x - 0.5) ** 2 for x in individual[3:]])

    f1 = (1 + g) * np.prod([np.cos(x * np.pi / 2) for x in individual[:3]])
    f2 = (1 + g) * np.prod([np.cos(x * np.pi / 2) for x in individual[:2]]) * np.sin(individual[2] * np.pi / 2)
    f3 = (1 + g) * np.sin(individual[0] * np.pi / 2)

    obj_result = np.array([f1, f2, f3])
    constr_result = np.array([])

    return obj_result, (constr_result > 0) * constr_result


def example_problem_2():
    """
    Define example problem2 2 with 3 decision variables, 3 objectives and no constraints based on DTLZ2.

    """
    problem2 = Problem(n_vars=3, n_objs=3, n_constrs=0)
    problem2.set_vars_bounds([0, 0, 0], [1, 1, 1])
    problem2.get_individual_result = ex2_get_individual_result

    return problem2


if __name__ == "__main__":
    """
    Run examples of NSGA II.

    """

    np.random.seed(123)  # Fix global random seed

    problem_list = [example_problem_1(), example_problem_2()]

    for i, problem in enumerate(problem_list, 1):
        print(f"Example problem {i}:")
        nsga2 = ConstrainedNSGA2(problem, pop_size=500, max_gen=1000)
        nsga2.run()
        print("\n")
