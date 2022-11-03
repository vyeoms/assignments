# Write your functions lp1() and lp2() to replace the functions here. Do not change the names of the functions.
# Then copy the content of this cell into a separate file named 'problem2.py' to be submitted separately on Moodle.
# The file should include these import instructions and no others.

import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg
import scipy.special as scs

def lp1(x, alpha, w):
    """Returns log p(x | \alpha, w).
    Input:
    x: double
    alpha: np.array, shape: (n,)
    w: np.array, shape: (n,)
    """
    log_denom = np.log(2) + scs.gammaln(alpha) # Compute the log of the denominator of the sum
    log_num = np.log(w) + np.log(np.power(np.abs(x), alpha-1)) - np.abs(x) # log of the numerator of the sum
    return scs.logsumexp(log_num - log_denom) # Exponentiate the log of the sum term, sum over all the terms, and return the log probability

# Helper function to calculate the log-determinant with the Cholesky decomposition
def cholLogDet(A):
    L = lg.cholesky(A)
    return 2*np.sum(np.log(L.diagonal()))

def lp2(x, V, n):
    """Returns log p(x | V, n)
    Input:
    x: np.array, shape: (d,d)
    V: np.array, shape: (d,d)
    n: double, n > d-1
    """
    p = x.shape[0] # Determine p
    
    # Calculate the explicit inverse of L with the Cholesky approach
    inv_chol_L = slg.solve_triangular(lg.cholesky(V), np.eye(p), lower=True)
    inv_V = np.matmul(inv_chol_L.T, inv_chol_L)
    
    # Calculate the log of the numerator and denominator
    log_num = 0.5*(n-p-1) * cholLogDet(x) - 0.5*np.einsum("ij, ij->", inv_V.T, x)
    log_denom = 0.5*n*p*np.log(2) + 0.5*n*cholLogDet(V) + scs.multigammaln(0.5*n, p)
    
    # Return the log probability
    return log_num - log_denom

