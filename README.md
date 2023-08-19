# Constrained NSGA2
This is an implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for solving multi-objective optimization problems with constraints. 
It includes a Problem class for defining optimization problems and the main ConstrainedNSGA2 class for solving them. 

## Usage
### Defining a Problem
To define a problem, create an instance of the Problem class and customize it according to your optimization problem. 
You can set the number of decision variables, objectives, and constraints, as well as variable bounds, linear objective functions, and linear constraint functions.
If your problem is non-linear, you will need to overwrite the get_individual_result() method to calculate the objective values and constraint violations for each individual in the population.

### Running NSGA-II
To solve an optimization problem using Constrained NSGA-II, create an instance of the ConstrainedNSGA2 class and pass in your problem instance as a parameter, along with any additional configuration options such as population size, maximum generations, and mutation or crossover probabilities. 
Then, call the run() method on your ConstrainedNSGA2 instance to execute the algorithm and obtain the Pareto front.

### Examples
The examples.py script includes example code for running NSGA-II on two example problems. 
The script demonstrates how to define custom optimization problems, create ConstrainedNSGA2 instances, and execute the algorithm to obtain and display the results.

## Reference

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE transactions on evolutionary computation*, *6*(2), 182-197.
2. Deb, K., Agrawal, S. (1999). A Niched-Penalty Approach for Constraint Handling in Genetic Algorithms. In: *Artificial Neural Nets and Genetic Algorithms*. Springer, Vienna.
