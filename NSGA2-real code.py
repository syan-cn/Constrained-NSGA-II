# -*- coding: utf-8 -*-
"""
Author: Yan Sen
Title: Constarined NSGA-II
"""
import time
time_start=time.time()
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

##Parameters
M = 3                  # Number of objectives
C = 4                  # Number of constraints
V = 2                  # Number of variables
pop_size = 200         # Population size
gen_max = 500          # Max number of generations - stopping criteria
xl = np.array([0,0])   # Lower bound vector
xu = np.array([10,10]) # Upper bound vector 
etac = 20              # Distribution index for crossover
etam = 100             # Distribution index for mutation / mutation constant
pc = 0.8               # Crossover Probability
pm=0.1                 # Mutation Probability
#np.random.seed(123)    # Fix global random seed

## Define functions
def problem(x):
    '''
    1.This function returns the objective functions f1, f2, ..., fM in the vector 'fit' and constraints in the vector 'c' for the chromosome 'x'. 
    2.All the constraints 'c' are converted to the form h(x)<=0.
    '''
    fit = np.zeros(M)
    c = np.zeros(C)
    fit[0] = -np.dot(np.array([2,3]),x.T)
    fit[1] = np.dot(np.array([3,4]),x.T)
    fit[2] = -np.dot(np.array([2,5]),x.T)
    c[0] = 3-np.dot(np.array([1,1]),x.T)
    c[1] = np.dot(np.array([1,1]),x.T)-6
    c[2] = 10-np.dot(np.array([2,3]),x.T)
    c[3] = np.dot(np.array([2,3]),x.T)-15
    error = (c>0)*c
    return fit, error

def normalisation(err):
    '''
    1.This function normalises the constraint violation of various individuals, since the range of constraint violation of every chromosome is not uniform.
    2.Input is in the matrix error_pop with size [pop_size, number of constraints]; Output is a normalised vector, err_norm of size [pop_size,1].
    '''
    #Error Nomalisation
    con_max=0.000000001+err.max(axis=0)
    con_maxx=np.tile(con_max,(pop_size,1))
    cc=err/con_maxx
    err_norm=cc.sum(axis=1)             # finally sum up all violations
    return err_norm

def NDS_CD_cons(population):
    '''
    1.This function is to perform Deb's fast elitist non-domination sorting and crowding distance assignment with constraints.
    2.Input is in the variable 'population' with size: [size(popuation), V+M+1]; This function returns 'chromosome_NDS_CD' with size [size(population),V+M+3].
    3.A flag 'problem_type' is used to identify whether the population is fully feasible (problem_type=0) or fully infeasible (problem_type=1) or partly feasible (problem_type=0.5).
    '''
    ## Initialising structures and variables
    front=[]
    rank = 0
    
    ## Segregating feasible and infeasible solutions
    if all(population[:,V+M]==0):
        problem_type = 0
        chromosome = population[:,:V+M]                         # All are feasible chromosomes
        pop_size1 = chromosome.shape[0]
    elif all(population[:,V+M]!=0):
        problem_type = 1
        pop_size1 = 0
        infchromosome = population                              # All are inFeasible chromosomes
    else:
        problem_type = 0.5
        feas_index = np.where(population[:,V+M]==0)[0]
        chromosome = population[feas_index,:V+M]                # Feasible chromosomes
        pop_size1 = chromosome.shape[0]
        infeas_index = np.where(population[:,V+M]!=0)[0]
        infchromosome = population[infeas_index,:]              # Infeasible chromosomes
    
    ## Handling feasible solutions 
    if problem_type==0 or problem_type==0.5:
        f = chromosome[:,V:V+M]                                 # Objective function values
        # Nondomination Sorting
        # Find domination relationship between solutions
        n_p = np.zeros(pop_size1)
        s_p = []
        for p in range(pop_size1):
            s_p.append(np.where(((f[p,:]-f[:,:]<=0).all(axis=1))&(~((f[p,:]-f[:,:]==0).all(axis=1))))[0])       # Chromosomes are dominated by chromosome p
            n_p[p] = len(np.where(((f[p,:]-f[:,:]>=0).all(axis=1))&(~((f[p,:]-f[:,:]==0).all(axis=1))))[0])     # Number of chromosomes dominate chromosome p
        # First front
        front.append(np.where(n_p==0)[0])
        # Creating subsequent fronts
        chromosome = np.column_stack((chromosome, np.zeros(pop_size1)))
        while len(front[rank])!=0:
            front_indiv = front[rank]
            n_p[front_indiv] = float('inf')
            chromosome[front_indiv,V+M] = rank
            rank += 1
            for i in range(len(front_indiv)):
                temp = s_p[front_indiv[i]]
                n_p[temp] -= 1
            front.append(np.where(n_p==0)[0])
        chromosome_sorted = chromosome[np.lexsort(chromosome.T)] # Ranked population
        # Crowding distance assignment
        rowsindex = 0
        chromosome_sorted = np.column_stack((chromosome_sorted, np.zeros([pop_size1,M])))
        for i in range(len(front)-1):
            l_f = len(front[i])
            if l_f>2:
                # Sorting based on function value
                sorted_ind = np.argsort(chromosome_sorted[rowsindex:(rowsindex+l_f),V:V+M], axis=0)
                fmin = np.zeros(M)
                fmax = np.zeros(M)
                for m in range(M):
                    fmin[m] = chromosome_sorted[sorted_ind[0,m]+rowsindex,V+m]
                    fmax[m] = chromosome_sorted[sorted_ind[-1,m]+rowsindex,V+m]
                    chromosome_sorted[sorted_ind[0,m]+rowsindex,V+M+m+1] = float('inf')
                    chromosome_sorted[sorted_ind[-1,m]+rowsindex,V+M+m+1] = float('inf')
                for j in range(1,l_f-1):
                    for m in range(M):
                        if fmax[m]-fmin[m]==0:
                            chromosome_sorted[sorted_ind[j,m]+rowsindex,V+M+m+1] = float('inf')
                        else:
                            chromosome_sorted[sorted_ind[j,m]+rowsindex,V+M+m+1] = (chromosome_sorted[sorted_ind[j+1,m]+rowsindex,V+m]-chromosome_sorted[sorted_ind[j-1,m]+rowsindex,V+m])/(fmax[m]-fmin[m])
            else:
                chromosome_sorted[rowsindex:(rowsindex+l_f),V+M+1:V+M+M+1] = float('inf')
            rowsindex += l_f
        chromosome_sorted = np.column_stack((chromosome_sorted, chromosome_sorted[:,V+M+1:V+M+M+1].sum(axis=1)))
        # Final output for feasible solutions: Variables, Objects, normalized Error(0), Rank, normalized Crowding distance
        chromosome_NDS_CD1 = np.column_stack((chromosome_sorted[:,:V+M],np.zeros([pop_size1,1]),chromosome_sorted[:,V+M],chromosome_sorted[:,V+M+M+1]))
    
    ## Handling infeasible solutions
    if problem_type==1 or problem_type==0.5:
        infpop = infchromosome[infchromosome[:,V+M].argsort()]
        infpop = np.column_stack((infpop[:,:V+M+1],np.array(range(rank,rank+infpop.shape[0])),float('inf')*(np.ones(infpop.shape[0]))))
        if problem_type==0.5:
            del front[-1]
        for i in range(len(infchromosome)):
            front.append(np.array([pop_size1+i]))
    
    ## Combine results
    if problem_type == 0:
        chromosome_NDS_CD = chromosome_NDS_CD1
    elif problem_type == 1:
        chromosome_NDS_CD = infpop
    else:
        chromosome_NDS_CD = np.row_stack((chromosome_NDS_CD1,infpop))
    return chromosome_NDS_CD, front

def tour_selection(pool):
    '''
    1.Parents are selected from the population pool for reproduction by using binary tournament selection based on the rank and crowding distance.
    2.An individual is selected if the rank is lesser than the other or if crowding distance is greater than the other.
    3.Input and output are of same size [pop_size, V+M+3].
    '''
    ## Binary tournament selection
    parent_selected = np.zeros([pop_size, V+M+3])
    rank_col = V+M+1
    distance_col = V+M+2
    candidate = np.random.randint(pop_size,size=(pop_size,2))
    for i in range(pop_size):
        parent = candidate[i,:]                                                # Two parents indexes are randomly selected
        if pool[parent[0],rank_col] < pool[parent[1],rank_col]:                # For parents with different ranks, check the rank of two individuals
            parent_selected[i,:] = pool[parent[0],:]                    # Minimum rank individual is selected finally
        elif pool[parent[0],rank_col] > pool[parent[1],rank_col]:
            parent_selected[i,:] = pool[parent[1],:]
        else:
            if pool[parent[0],distance_col] > pool[parent[1],distance_col]:    # For parents with same ranks, check the distance of two parents
                parent_selected[i,:] = pool[parent[0],:]                       # Maximum distance individual is selected finally
            elif pool[parent[0],distance_col] < pool[parent[1],distance_col]:
                parent_selected[i,:] = pool[parent[1],:]
            else:
                parent_selected[i,:] = pool[parent[np.random.randint(2)],:]
    return parent_selected

def poly_mutation(y):
    '''
    1.Input is the crossovered child of size (1,V) in the vector 'y' from function 'genetic_operator'.
    2.Output is in the vector 'mutated_child' of size (1,V).
    '''
    ## Polynomial mutation including boundary constraint
    de = np.row_stack((y-xl,xu-y)).min(0)/(xu-xl)
    t = np.random.random(V)
    loc_mut = (t<pm)
    u = np.random.random(V)
    deq=np.zeros(V)
    for i in range(len(y)):
        if u[i]<=0.5:
            deq[i]=np.power(((2*u[i])+((1-2*u[i])*np.power((1-de[i]),(etam+1)))),(1/(etam+1)))-1
        elif u[i]>0.5:
            deq[i]=1-np.power(((2*(1-u[i]))+(2*(u[i]-0.5)*np.power((1-de[i]),(etam+1)))),(1/(etam+1)))
#    deq = (u<=0.5)*(np.power(((2*u)+((1-2*u)*np.power((1-de),(etam+1)))),(1/(etam+1)))-1)+(u>0.5)*(1-np.power(((2*(1-u))+(2*(u-0.5)*np.power((1-de),(etam+1)))),(1/(etam+1))))
    mutated_child = y+deq*loc_mut*(xu-xl)
    return mutated_child

def genetic_operator(parent_selected):
    '''
    1.Crossover followed by mutation.
    2.Input is in 'parent_selected' matrix of size [pop_size,V].
    3.Output is also of same size in 'child_offspring'.
    '''
    ## SBX cross over operation incorporating boundary constraint
    # Reference: Deb & samir agrawal,"A Niched-Penalty Approach for Constraint Handling in Genetic Algorithms".
    rc = np.random.randint(pop_size,size=pop_size)
    child_offspring = np.zeros([pop_size,V])
    for i in range(int(pop_size/2)):
        if np.random.random() <= pc:
            parent1 = parent_selected[rc[2*i],:]
            parent2 = parent_selected[rc[2*i+1],:]
            rnd = np.random.random(V)
            child1 = np.zeros(V)
            child2 = np.zeros(V)
            for j in range(V):
                if parent1[j] == parent2[j] or rnd[j] > 0.5:
                    child1[j] = parent1[j]
                    child2[j] = parent2[j]
                else:
                    beta = 1+2*min((parent1[j]-xl[j]),(xu[j]-parent2[j]))/abs(parent2[j]-parent1[j])
                    alpha = 2-pow(beta, -(etac+1))
                    u = np.random.random()
                    betaq = (u<=(1/alpha))*pow(u*alpha,1/(etac+1))+(u>(1/alpha))*pow(1/(2-u*alpha),1/(etac+1))
                    child1[j] = 0.5*((1+betaq)*parent1[j]+(1-betaq)*parent2[j])
                    child2[j] = 0.5*((1-betaq)*parent1[j]+(1+betaq)*parent2[j])
        else:
            child1 = parent_selected[rc[2*i],:]
            child2 = parent_selected[rc[2*i+1],:]
        # polynomial mutation
        child_offspring[rc[2*i],:] = poly_mutation(child1)
        child_offspring[rc[2*i+1],:] = poly_mutation(child2)
    return child_offspring

def replacement(population_inter_sorted, front):
    '''
    The next generation population is formed by appending each front subsequently until the
    population size exceeds the current population size. If When adding all the individuals
    of any front, the population exceeds the population size, then the required number of
    remaining individuals alone are selected from that particular front based on crowding distance.
    '''
    new_pop = np.zeros([pop_size, population_inter_sorted.shape[1]])
    index=0
    i=0
    while index < pop_size:
        l_f = len(front[i])
        if index+l_f < pop_size:
            new_pop[index:index+l_f,:] = population_inter_sorted[index:index+l_f,:]
            index += l_f
        else:
            temp1 = population_inter_sorted[index:index+l_f,:]
            temp2 = temp1[temp1[:,-1].argsort()]
            new_pop[index:pop_size,:] = temp2[l_f-(pop_size-index):l_f,:]
            index += l_f
        i += 1
    return new_pop

if __name__=="__main__":
    # 1.This is the main program of NSGA II.
    # 2.This code defines population size in 'pop_size', number of design variables in 'V', 
    #   maximum number of generations in 'gen_max', current generation in 'gen_count' and number of objectives in 'M'.
    # 3.Final optimal Pareto soutions are in the variable 'pareto_rank1', with design variables in the coumns (1:V),
    #   objectives in the columns (V+1 to V+M), constraint violation in the column (V+M+1), Rank in (V+M+2), Distance in (V+M+3).
    
    ## Initial population
    xl_temp = np.tile(xl,(pop_size,1))
    xu_temp = np.tile(xu,(pop_size,1))
    x = xl_temp+(xu_temp-xl_temp)*np.random.random((pop_size,V))
    
    ## Evaluate objective function and constraints
    ff = np.zeros([pop_size,M])
    err = np.zeros([pop_size,C])
    for i in range(pop_size):
	    ff[i,:], err[i,:] = problem(x[i,:])
    error_norm = normalisation(err)                                                    # Normalisation of the constraint violation
    population_init = np.concatenate((x, ff, error_norm.reshape(pop_size,1)), axis=1)
    population, front = NDS_CD_cons(population_init)                                   # Non domination Sorting on initial population
    
    ##Generation starts
    for gen_count in range(gen_max):
        print(gen_count)
        # Selection (Parent Pt of 'N' pop size)
        parent_selected = tour_selection(population)                                   # Binary Tournament Selection
        # Reproduction (Offspring Qt of 'N' pop size)
        child_offspring = genetic_operator(parent_selected[:,:V])                      # SBX crossover and polynomial mutation
        # Objective function and constraints evaluation for offspring
        fff = np.zeros([pop_size,M])
        err = np.zeros([pop_size,C])
        for i in range(pop_size):
            fff[i,:], err[i,:] = problem(child_offspring[i,:])
        error_norm = normalisation(err) 
        child_offspring = np.concatenate((child_offspring, fff, error_norm.reshape(pop_size,1)), axis=1)
        # Intermediate population ( Rt=(Pt)U(Qt) of 2N size)
        population_inter = np.row_stack((population[:,:V+M+1], child_offspring[:,:V+M+1]))
        population_inter_sorted, front = NDS_CD_cons(population_inter)                 # Non domination Sorting on offspring
        # Replacement - N
        population = replacement(population_inter_sorted, front)
    
    ## Result and Pareto plot
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y, z = population[:,V], population[:,V+1], population[:,V+2]
    margin = 0.1
    ax.scatter(x, y, z, s=10, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('My problem')
    ax.view_init(30, 35)
    plt.show()

time_end=time.time()  
print(time_end-time_start) 