# NSGA-II
A Python code for NSGA-II algorithm (real code) is developed for a test problem in this file (Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197).
This code can solve multi-objective programming of any number of objects with constraints. Before using, you need to modify the "problem" function and corresponding parameters.
In this code, SBX crossover operation incorporating boundary constraint and polynomial mutation including boundary constraint are used. And in the nondomination sorting and calculation of crowding distance, the nomalisation value of constraint violation is considered.
