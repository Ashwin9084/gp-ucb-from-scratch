import numpy as np 
import matplotlib.pyplot as plt

class GPBandit:
    def __init__(self, n_points = 100, noise_std=0.1, seed=0):
        
        ## Setting the seed values
        self.rng = np.random.default_rng(seed)
        np.random.seed(0)
        np.set_printoptions(precision=3)
        

        ## Making then inputs and the hiddden function
        self.X = np.linspace(0,5, n_points)
        self.noise_std = noise_std
        self.n = n_points


        ## just making the hidden function sum of two nice guassains
        tall  = 1.0 * np.exp(-(self.X - 3.5)**2 / (2 * 0.3**2))
        short = 0.6 * np.exp(-(self.X - 1.0)**2 / (2 * 0.6**2))
        self.f_values = tall + short

        self.f_star = np.max(self.f_values) ## The best value we got

    
    
    def generate_reward(self, i):
        noise = self.rng.normal(0,self.noise_std)
        reward = self.f_values[i] + noise
        return reward

if __name__ == "__main__":
    bandit = GPBandit()
    plt.plot(bandit.X, bandit.f_values)
    plt.show()