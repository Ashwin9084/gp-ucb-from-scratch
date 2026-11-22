import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from bandits import GPBandit
from solvers import EpsilonGreedy, UCB1, GPUCB, GPThompson
from helpers import plot_results, report



def experiment(K,N):
    '''
    K bandits and N tries 
    '''

    b = GPBandit(K, seed=0)
    # print(b.best_prob, b.probs, b.n)

    solver_egreedy = EpsilonGreedy(b, 1)
    solver_UCB1 = UCB1(b)
    solver_gpucb = GPUCB(b)
    solver_gpthompson= GPThompson(b)

    solver_egreedy.run(N)
    report(solver_egreedy, "Epsilon_greedy")

    solver_UCB1.run(N)
    report(solver_UCB1, "UCB1")

    solver_gpucb.run(N)
    report(solver_gpucb, "GPUCB")

    solver_gpthompson.run(N)
    report(solver_gpthompson, "GPThompson")

    solvers = [solver_egreedy, solver_UCB1, solver_gpucb, solver_gpthompson]
    names = ["Random (eps=1)", "UCB1", "GP-UCB", "GP-Thompson"]
    plot_results(solvers, solver_names=names)

if __name__=="__main__":
    ## K is the number of points in our grid and N are the number of tries
    experiment(K=100, N=500)