import numpy as np
from numpy.linalg import inv


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

X = np.arange(-5,5,0.2).reshape(-1,1)



def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    
    X_s = X_s.reshape(-1,1)
    X_train = X_train.reshape(-1,1)
    Y_train = Y_train.reshape(-1,1)
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n)
    """
    
    k = kernel(X_train, X_train, l, sigma_f) + sigma_y**2  * np.eye(len(X_train))
    k_s = kernel(X_train, X_s, l, sigma_f)
    k_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8*np.eye(len(X_s)) ## Noise to  make sure we do not get a zero value
    k_inv = inv(k)


    # Equation (7)  ## link: https://web.archive.org/web/20250516052342/https://nbviewer.org/github/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes.ipynb?flush_cache=true
    mu_s = k_s.T.dot(k_inv).dot(Y_train)

    # Equation (8)
    cov_s = k_ss - k_s.T.dot(k_inv).dot(k_s)
    
    return mu_s, cov_s
