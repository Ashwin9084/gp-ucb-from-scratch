
import numpy as np
from gp import *

class solver():
    def __init__(self, bandit):


        
        self.bandit = bandit 
        self.counts = [0] * self.bandit.n
        self.actions = [] # List of machine choices denoted by thier respective indices
        self.regret = 0 # How far away we are from the best action :(
        self.regrets = []
        self.recommendations = []

    def update_regret(self, i):
        prob_i = self.bandit.f_values[i]
        self.regret += (self.bandit.f_star - prob_i)
        self.regrets.append(self.regret) 

    def run_one_step(self):
        pass

    def recommend(self):
        pass

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)
            self.recommendations.append(self.recommend())


class EpsilonGreedy(solver):
    def __init__(self, bandit, eps, init_proba = 1.0):


        super().__init__(bandit)
        self.eps = eps
        self.int_proba = init_proba

        self.estimates = [self.int_proba] * self.bandit.n

    def recommend(self): ## only for plotting can be ignored for undestanding the algorithm
        pulled = np.array(self.counts) > 0
        scores = np.where(pulled, self.estimates, -np.inf)
        return int(np.argmax(scores))

    def run_one_step(self):
        if np.random.random() < self.eps:
            ## Let's do exploration
            i = np.random.randint(0, self.bandit.n)
        else:
            ## Let's continue with some exploitation
            i = np.argmax(self.estimates)

        r = self.bandit.generate_reward(i)


        ## this does step wisse average in the form of new = old + error*(1/stepsize)
        self.estimates[i] +=1/(self.counts[i] + 1)*(r - self.estimates[i]) 
        '''
        Alternatively this also works 
        self.rewards[i] += r
        self.estimates = self.rewards[i]/(self.counts[i] + 1)
        '''

        return i

class UCB1(solver):
    def __init__(self, bandit, init_proba = 1.0):


        super().__init__(bandit)
        self.t = 0
        self.int_proba = init_proba

        self.estimates = [self.int_proba] * self.bandit.n 

    def recommend(self): ## only for plotting can be ignored for undestanding the algorithm
        pulled = np.array(self.counts) > 0
        scores = np.where(pulled, self.estimates, -np.inf)
        return int(np.argmax(scores))

    def run_one_step(self):
        self.t +=1 
        hoff_inequality = [self.estimates[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x])) for x in range(self.bandit.n)]

        i = np.argmax(hoff_inequality)

        r = self.bandit.generate_reward(i)


        ## this does step wisse average in the form of new = old + error*(1/stepsize)
        self.estimates[i] +=1/(self.counts[i] + 1)*(r - self.estimates[i]) 
        '''
        Alternatively this also works 
        self.rewards[i] += r
        self.estimates = self.rewards[i]/(self.counts[i] + 1)
        '''

        return i


class GPThompson(solver):
    def __init__(self, bandit, ell=0.5, sigma_f=1.0, sigma_y=0.1):
        super().__init__(bandit)

        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        
        self.ell = ell

        self.X_seen = []
        self.r_seen = [] 

        ##for ploting
        self.estimates = [0]* self.bandit.n 

    def recommend(self):
        return int(np.argmax(self.estimates))      # estimates is the posterior mean

    def run_one_step(self):
        if not self.X_seen:
            i = np.random.randint(0, self.bandit.n)
        else:
            X = self.bandit.X
            X_train = np.array(self.X_seen)
            Y_train = np.array(self.r_seen)
            mu_s, cov_s = posterior(X_s=X, X_train=X_train, Y_train=Y_train, l=self.ell, sigma_f=self.sigma_f, sigma_y=self.sigma_y)
            # sigma = np.sqrt(np.diag(cov_s)).reshape(-1,1)
            # samples = mu_s + self.beta*sigma

            samples = np.random.multivariate_normal(mu_s.ravel(), cov_s)
            i = np.argmax(samples)
            


        r = self.bandit.generate_reward(i)
        self.X_seen.append(self.bandit.X[i])
        self.r_seen.append(r)

        # This is just for making sure, the plots work 
        if len(self.X_seen)>1: self.estimates = mu_s.ravel()
        

        return i



class GPUCB(solver):
    def __init__(self, bandit, ell=0.5, sigma_f=1.0, sigma_y=0.1, beta=2.0):
        super().__init__(bandit)

        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.beta = beta
        self.ell = ell

        self.X_seen = []
        self.r_seen = [] 

        ##for ploting
        self.estimates = [0]* self.bandit.n 

    def recommend(self):
        return int(np.argmax(self.estimates))      # estimates is the posterior mean

    def run_one_step(self):
        if not self.X_seen:
            i = np.random.randint(0, self.bandit.n)
        else:
            X = self.bandit.X
            X_train = np.array(self.X_seen)
            Y_train = np.array(self.r_seen)
            mu_s, cov_s = posterior(X_s=X, X_train=X_train, Y_train=Y_train, l=self.ell, sigma_f=self.sigma_f, sigma_y=self.sigma_y)
            sigma = np.sqrt(np.diag(cov_s)).reshape(-1,1)
            samples = mu_s + self.beta*sigma
            i = np.argmax(samples)
            


        r = self.bandit.generate_reward(i)
        self.X_seen.append(self.bandit.X[i])
        self.r_seen.append(r)

        # This is just for making sure, the plots work 
        if len(self.X_seen)>1: self.estimates = mu_s.ravel()
        

        return i