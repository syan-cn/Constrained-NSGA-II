# -*- coding: utf-8 -*-
# @author: YanSen
# @date: 2023/05/03

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from problem import Problem


class ConstrainedNSGA2(object):
    """
    Constrained Non-dominated Sorting Genetic Algorithm II (NSGA-II)

    Parameters:
        problem: an instance of the Problem class representing the optimization problem
        pop_size: the size of the population
        max_gen: the maximum number of generations (stopping criteria)
        etac: distribution index for crossover
        etam: distribution index for mutation (mutation constant)
        pc: crossover probability
        pm: mutation probability
        if_plot_front: boolean flag indicating whether to plot the Pareto front or not
    """

    def __init__(
            self,
            problem: Problem,
            pop_size,
            max_gen,
            etac=20,
            etam=100,
            pc=0.8,
            pm=0.1,
            if_plot_front=True
    ):
        self._problem = problem
        self._n_objs = problem.n_objs
        self._n_vars = problem.n_vars
        self._n_cons = problem.n_constrs
        self._vars_lb = problem.vars_lb
        self._vars_ub = problem.vars_ub
        self._pop_size = pop_size
        self._max_gen = max_gen
        self._etac = etac
        self._etam = etam
        self._pc = pc
        self._pm = pm
        self._if_plot_front = if_plot_front

    def run(self):
        """
        Run the NSGA-II algorithm for the given problem and parameters.

        """
        population = self._initialize_population()

        # Start the generation loop
        for _ in tqdm(range(self._max_gen), desc="Generation Progress",
                      bar_format="{desc}: {n}/{total} |{bar}| {percentage:3.0f}% "
                                 "Elapsed: {elapsed} Remaining: {remaining} Rate: {rate_fmt}{postfix}",
                      ncols=100, unit=" gen"):
            # Select the parent population using binary tournament selection
            selected_parent_pop = self._tournament_selection(population)  # Binary Tournament Selection
            # Select the parent population using binary tournament selection
            child_pop = self._genetic_operator(selected_parent_pop[:, :self._n_vars])
            # Evaluate the child population
            pop_norm_violation, pop_objs = self._evaluate_population(child_pop)
            child_pop = np.concatenate(
                (child_pop, pop_objs, pop_norm_violation.reshape(self._pop_size, 1)), axis=1)
            # Combine the original and child populations
            comb_pop = np.row_stack(
                (population[:, :self._n_vars + self._n_objs + 1], child_pop[:, :self._n_vars + self._n_objs + 1]))
            sorted_inter_pop, front = self._non_dominated_sorting_and_crowding_distance(comb_pop)
            # Replace the original population with the new population for the next iteration
            population = self._replacement(sorted_inter_pop, front)

        # Plot the Pareto front if specified
        if self._if_plot_front:
            self._plot_pareto_front(population)

        return population

    def _plot_pareto_front(self, population):
        """
        Plot the Pareto front for the given population.

        Parameters:
            population: the population of solutions

        """

        plt.figure(figsize=(10, 8))

        if self._n_objs == 2:
            x, y = population[:, self._n_vars], population[:, self._n_vars + 1]
            plt.scatter(x, y, s=50, c='blue', marker='.', edgecolors='darkgray', alpha=0.8, label='Pareto Front')
            plt.xlabel('Objective 1', fontsize=12)
            plt.ylabel('Objective 2', fontsize=12)
            plt.title('Pareto Front', fontsize=16)
        elif self._n_objs == 3:
            ax = plt.gca(projection='3d')
            x, y, z = population[:, self._n_vars], population[:, self._n_vars + 1], population[:, self._n_vars + 2]
            ax.scatter(x, y, z, s=50, c='blue', marker='.', edgecolors='darkgray', alpha=0.8, label='Pareto Front')
            ax.set_xlabel('Objective 1', fontsize=12, labelpad=10)
            ax.set_ylabel('Objective 2', fontsize=12, labelpad=10)
            ax.set_zlabel('Objective 3', fontsize=12, labelpad=10)
            ax.view_init(elev=30, azim=35)
        else:
            print("Plotting for more than 3 objectives is not supported.")
            return

        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.legend(loc='upper left', fontsize=12)
        plt.show()

    def _initialize_population(self):
        """
        Initialize the population based on the upper and lower bounds of variables, and evaluate the population.

        """
        pop_vars_lb = np.tile(self._vars_lb, (self._pop_size, 1))
        pop_vars_ub = np.tile(self._vars_ub, (self._pop_size, 1))
        init_pop_vars = pop_vars_lb + (pop_vars_ub - pop_vars_lb) * np.random.random((self._pop_size, self._n_vars))

        pop_norm_violation, pop_objs = self._evaluate_population(init_pop_vars)
        population_init = np.concatenate((init_pop_vars, pop_objs, pop_norm_violation.reshape(self._pop_size, 1)),
                                         axis=1)
        population, front = self._non_dominated_sorting_and_crowding_distance(population_init)
        return population

    def _evaluate_population(self, pop_vars):
        """
        Evaluate the objectives and normalize the constraint violations for the population.

        Parameters:
            pop_vars: The population variables to be evaluated.

        """
        pop_objs = np.zeros([self._pop_size, self._n_objs])
        pop_violation = np.zeros([self._pop_size, self._n_cons])
        for ind_idx in range(self._pop_size):
            pop_objs[ind_idx, :], pop_violation[ind_idx, :] = self._problem.get_individual_result(pop_vars[ind_idx, :])
        pop_norm_violation = self.__normalize_constraint_violation(pop_violation)
        return pop_norm_violation, pop_objs

    @staticmethod
    def __normalize_constraint_violation(pop_violation):
        """
        Normalize the constraint violations of various individuals, as the range of constraint violation for each
        chromosome is not uniform.

        Parameters:
            pop_violation: The constraint violations of the population.

        """

        violation_max = pop_violation.max(axis=0)
        normalized_violations = np.divide(pop_violation, violation_max, out=np.ones_like(pop_violation) * np.inf,
                                          where=violation_max != 0)
        normalized_violation_sum = normalized_violations.sum(axis=1)
        return normalized_violation_sum

    def _tournament_selection(self, original_pop):
        """
        Parents are selected from the population pool for reproduction using binary tournament selection based on
        rank and crowding distance.

        Parameters:
            original_pop: The original population.

        """
        rank_col = self._n_vars + self._n_objs + 1
        dist_col = self._n_vars + self._n_objs + 2

        candidate_pair = np.random.randint(self._pop_size, size=(self._pop_size, 2))

        rank_diff = original_pop[candidate_pair[:, 0], rank_col] - original_pop[candidate_pair[:, 1], rank_col]

        cand_1_dist = original_pop[candidate_pair[:, 1], dist_col]
        cand_0_dist = original_pop[candidate_pair[:, 0], dist_col]
        dist_diff = np.subtract(cand_1_dist, cand_0_dist, out=np.zeros_like(cand_0_dist),
                                where=((cand_0_dist != np.inf) & (cand_1_dist != np.inf)))

        selected_indices = np.where(
            (rank_diff < 0) | ((rank_diff == 0) & (dist_diff > 0)), candidate_pair[:, 0], candidate_pair[:, 1])
        result_pop = original_pop[selected_indices, :]
        return result_pop

    def _genetic_operator(self, selected_parent_pop):
        """
        Perform crossover followed by mutation.
        Reference: "A Niched-Penalty Approach for Constraint Handling in Genetic Algorithms".

        Parameters:
            selected_parent_pop: The selected parent population.

        """
        rand_parent_indices = np.random.randint(self._pop_size, size=self._pop_size)
        result_pop = np.zeros([self._pop_size, self._n_vars])
        for row_idx in range(int(self._pop_size / 2)):
            parent1 = selected_parent_pop[rand_parent_indices[2 * row_idx], :]
            parent2 = selected_parent_pop[rand_parent_indices[2 * row_idx + 1], :]
            child1, child2 = self.__simulated_binary_crossover(parent1, parent2)

            result_pop[rand_parent_indices[2 * row_idx], :] = self.__polynomial_mutation(child1)
            result_pop[rand_parent_indices[2 * row_idx + 1], :] = self.__polynomial_mutation(child2)

        return result_pop

    def __simulated_binary_crossover(self, parent1, parent2):
        """
        Perform simulated binary crossover (SBX) between two parents incorporating boundary constraint.

        Parameters:
            parent1: The first parent individual.
            parent2: The second parent individual.

        """
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        if np.random.random() <= self._pc:
            rnd = np.random.random(self._n_vars)
            for var_idx in range(self._n_vars):
                if parent1[var_idx] == parent2[var_idx] or rnd[var_idx] > 0.5:
                    continue

                if parent1[var_idx] < parent2[var_idx]:
                    parent1_var_value, parent2_var_value = parent1[var_idx], parent2[var_idx]
                else:
                    parent2_var_value, parent1_var_value = parent1[var_idx], parent2[var_idx]

                beta = 1 + 2 * min(
                    (parent1_var_value - self._vars_lb[var_idx]), (self._vars_ub[var_idx] - parent2_var_value)
                ) / abs(parent2_var_value - parent1_var_value)
                alpha = 2 - pow(beta, -(self._etac + 1))
                rand_cross_weight = np.random.random()
                if rand_cross_weight <= (1 / alpha):
                    cross_value = pow(rand_cross_weight * alpha, 1 / (self._etac + 1))
                else:
                    cross_value = pow(max(0, 1 / (2 - rand_cross_weight * alpha)), 1 / (self._etac + 1))

                child1[var_idx] = 0.5 * (parent1_var_value + parent2_var_value
                                         - cross_value * abs(parent2_var_value - parent1_var_value))
                child2[var_idx] = 0.5 * (parent1_var_value + parent2_var_value
                                         + cross_value * abs(parent2_var_value - parent1_var_value))

        return child1, child2

    def __polynomial_mutation(self, original_individual):
        """
        Perform polynomial mutation on a given individual incorporating boundary constraint.

        Parameters:
            original_individual: The original individual to be mutated.

        """
        # Polynomial mutation including boundary constraint
        norm_ind = np.minimum(original_individual - self._vars_lb,
                              self._vars_ub - original_individual) / (self._vars_ub - self._vars_lb)

        rand_delta_weight = np.random.random(self._n_vars)
        low_delta_loc, high_delta_loc = rand_delta_weight <= 0.5, rand_delta_weight > 0.5

        delta_value = np.zeros_like(norm_ind)
        delta_value[low_delta_loc] = np.power(
            ((2 * rand_delta_weight[low_delta_loc]) + ((1 - 2 * rand_delta_weight[low_delta_loc]) * np.power(
                (1 - norm_ind[low_delta_loc]), (self._etam + 1)))), (1 / (self._etam + 1))) - 1
        delta_value[high_delta_loc] = 1 - np.power(
            ((2 * (1 - rand_delta_weight[high_delta_loc])) +
             (2 * (rand_delta_weight[high_delta_loc] - 0.5) * np.power(
                 (1 - norm_ind[high_delta_loc]), (self._etam + 1)))), (1 / (self._etam + 1)))

        rand_loc_weight = np.random.random(self._n_vars)
        mutated_loc = (rand_loc_weight < self._pm)

        result_ind = original_individual + delta_value * mutated_loc * (self._vars_ub - self._vars_lb)

        return result_ind

    def _non_dominated_sorting_and_crowding_distance(self, population: np.ndarray) -> tuple:
        """
        Perform Deb's fast elitist non-domination sorting and crowding distance assignment with constraints.

        Parameters:
            population: An array of shape (pop_size, n_vars+n_objs+1) where the last column is a binary flag
            indicating whether the individual is feasible (0) or infeasible (1).

        Returns:
            tuple: A tuple containing the feasible population result and a list of fronts.

        """
        # Initialize result_pop
        result_pop = np.empty((0, self._n_vars + self._n_objs + 3))

        # Segregate feasible and infeasible solutions
        feasible_pop_mask = population[:, self._n_vars + self._n_objs] == 0

        feasible_pop = population[feasible_pop_mask, :self._n_vars + self._n_objs] if feasible_pop_mask.any() else None
        infeasible_pop = population[~feasible_pop_mask, :] if ~feasible_pop_mask.all() else None

        problem_type = 0 if feasible_pop_mask.all() else 1 if ~feasible_pop_mask.any() else 0.5
        feasible_pop_size = feasible_pop.shape[0] if feasible_pop is not None else 0

        # Initialize front list and rank
        fronts_result = []
        rank = 0

        # Handle feasible solutions
        if problem_type <= 0.5:
            feasible_pop_objs = feasible_pop[:, self._n_vars:self._n_vars + self._n_objs]

            # Perform non-domination sorting
            dominated_by = []
            dominates_count = np.zeros(feasible_pop_size)

            for ind_idx in range(feasible_pop_size):
                dominated_by.append(np.where(
                    ((feasible_pop_objs[ind_idx, :] - feasible_pop_objs[:, :] <= 0).all(axis=1))
                    & (~(feasible_pop_objs[ind_idx, :] - feasible_pop_objs[:, :] == 0).all(axis=1)))[0])
                dominates_count[ind_idx] = len(np.where(
                    ((feasible_pop_objs[ind_idx, :] - feasible_pop_objs[:, :] >= 0).all(axis=1))
                    & (~(feasible_pop_objs[ind_idx, :] - feasible_pop_objs[:, :] == 0).all(axis=1)))[0])

            # First front
            fronts_result.append(np.where(dominates_count == 0)[0])

            # Creating subsequent fronts_result
            feasible_pop = np.column_stack((feasible_pop, np.zeros(feasible_pop_size)))
            while len(fronts_result[rank]) != 0:
                front = fronts_result[rank]
                dominates_count[front] = np.inf
                feasible_pop[front, self._n_vars + self._n_objs] = rank
                rank += 1

                for ind_idx in range(len(front)):
                    dominates_count[dominated_by[front[ind_idx]]] -= 1

                fronts_result.append(np.where(dominates_count == 0)[0])

            # Sort feasible population based on ranks
            sorted_feasible_pop = feasible_pop[np.lexsort(feasible_pop.T)]

            # Assign crowding distance to feasible solutions
            sorted_feasible_pop = np.column_stack(
                (sorted_feasible_pop, np.zeros([feasible_pop_size, self._n_objs])))

            row_idx = 0
            for front_idx, front in enumerate(fronts_result[:-1]):
                start, end = row_idx, row_idx + len(front)
                crowding_distance = self.__calculate_crowding_distance(
                    sorted_feasible_pop[start:end, self._n_vars:self._n_vars + self._n_objs])
                sorted_feasible_pop[start:end, self._n_vars + self._n_objs + 1: self._n_vars + self._n_objs * 2 + 1] = \
                    crowding_distance
                row_idx += len(front)

            sorted_feasible_pop = np.column_stack(
                (sorted_feasible_pop,
                 sorted_feasible_pop[:, self._n_vars + self._n_objs + 1: self._n_vars + self._n_objs + self._n_objs + 1].sum(
                     axis=1)))

            # feasible solutions output: variables, objectives, normalized Error(0), Rank, normalized crowding distance
            feasible_pop_result = np.column_stack(
                (sorted_feasible_pop[:, :self._n_vars + self._n_objs], np.zeros([feasible_pop_size, 1]),
                 sorted_feasible_pop[:, self._n_vars + self._n_objs],
                 sorted_feasible_pop[:, self._n_vars + self._n_objs + self._n_objs + 1]))

            result_pop = np.row_stack((result_pop, feasible_pop_result))

        # Handle infeasible solutions
        if problem_type >= 0.5:
            infeasible_pop = infeasible_pop[infeasible_pop[:, self._n_vars + self._n_objs].argsort()]
            infeasible_pop = np.column_stack((
                infeasible_pop[:, :self._n_vars + self._n_objs + 1],
                np.array(range(rank, rank + infeasible_pop.shape[0])),
                np.inf * (np.ones(infeasible_pop.shape[0]))
            ))
            if problem_type == 0.5:
                del fronts_result[-1]
            for ind_idx in range(len(infeasible_pop)):
                fronts_result.append(np.array([feasible_pop_size + ind_idx]))

            result_pop = np.row_stack((result_pop, infeasible_pop))

        return result_pop, fronts_result

    def __calculate_crowding_distance(self, front_objs: np.array):
        """
        Calculate the crowding distance based on the objective values of the front.

        Parameters:
            front_objs: Array of objective values for the given front.

        """

        front_size = front_objs.shape[0]

        if front_size <= 2:
            return np.full((front_size, self._n_objs), np.inf)

        else:
            original_indices = np.argsort(front_objs, axis=0)
            sorted_front_objs = front_objs[original_indices, np.arange(self._n_objs)]

            obj_ranges = sorted_front_objs[-1] - sorted_front_objs[0]
            obj_ranges[obj_ranges == 0] = np.nan

            obj_distance = sorted_front_objs[1:, :] - sorted_front_objs[:-1, :]
            distance_to_last = np.row_stack([np.full(self._n_objs, np.inf), obj_distance])
            distance_to_next = np.row_stack([obj_distance, np.full(self._n_objs, np.inf)])

            obj_distance = (distance_to_last + distance_to_next) / obj_ranges

            restored_original_indices = np.argsort(original_indices, axis=0)
            result = obj_distance[restored_original_indices, np.arange(self._n_objs)]

        return result

    def _replacement(self, population, fronts):
        """
        The next generation population is formed by appending each front subsequently until the population size exceeds
        the current population size. If adding all the individuals of any front, the population exceeds the population
        size, then the required number of remaining individuals alone are selected from that particular front
        based on crowding distance.

        Parameters:
            population: The current population.
            fronts: List of fronts.

        """
        result_pop = np.zeros([self._pop_size, population.shape[1]])
        current_size = 0
        for _, front_pop in enumerate(fronts):
            front_size = len(front_pop)
            if current_size + front_size <= self._pop_size:
                result_pop[current_size:current_size + front_size, :] = \
                    population[current_size:current_size + front_size, :]
                current_size += front_size
            else:
                remain_ind = population[current_size:current_size + front_size, :]
                remain_ind_sorted = remain_ind[remain_ind[:, -1].argsort()]
                result_pop[current_size:self._pop_size, :] = \
                    remain_ind_sorted[front_size - (self._pop_size - current_size):front_size, :]
                break

        return result_pop
