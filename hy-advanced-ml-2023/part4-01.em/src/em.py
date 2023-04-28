import sys
import numpy as np


def logsumrows(X):
	"""
	Computes the sums of rows of log-numbers

	Parameters
	----------
    X : an array of size (n, m)
        matrix of log-numbers

	Returns
	-------
	s : an array of size n
		ith element is the sum of the ith row
	"""

	# place your code here
	n = X.shape[0]
	M = np.max(X, axis=1)
	return M + np.log(np.sum(np.exp(X - M.reshape((n, 1))), axis=1))


def computeparameters(R, X):
	"""
	Computes the optimal parameters for the Gaussian mixture model
	
	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
    R : an array of size (n, k)
		responsibilities: R[i, j] = probability of data point i belonging to
		jth component.

	Returns
	-------
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		optimal means: mu[i, :] is the optimal mean of the ith component
	C : an array of size (k, m, m)
		optimal covariances: C[i, :, :] is the optimal covariance matrix of the
		ith component
	"""

	k = R.shape[1]
	n, dim = X.shape

	prior = np.zeros(k)
	mu = np.zeros((k, dim))
	C = np.zeros((k, dim, dim))

	# place your code here
	for i in range(k):
		sum_Ri = np.sum(R[:,i])
		prior[i] = sum_Ri/n
		mu[i,:] = np.sum(R[:, i].reshape((n,1))*X, axis=0) / sum_Ri
		C[i,:,:] = np.cov(X - mu[i,:], aweights=R[:,i], ddof=0, rowvar=False)

	return prior, mu, C


def computeparametersdiagonal(R, X):
	"""
	Computes the optimal parameters for the Gaussian mixture model with
	diagonal covariance matrices.
	
	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
    R : an array of size (n, k)
		responsibilities: R[i, j] = probability of data point i belonging to
		jth component.

	Returns
	-------
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		optimal means: mu[i, :] is the optimal mean of the ith component
	C : an array of size (k, m, m)
		optimal covariances: C[i, :, :] is the optimal covariance diagonal
		matrix of the ith component 
	"""

	k = R.shape[1]
	n, dim = X.shape

	prior = np.zeros(k)
	mu = np.zeros((k, dim))
	C = np.zeros((k, dim, dim))

	# place your code here
	for i in range(k):
		s = np.zeros(dim)
		sum_Ri = np.sum(R[:, i])
		prior[i] = sum_Ri/n
		mu[i,:] = np.sum(R[:, i].reshape((n,1))*X, axis=0) / sum_Ri
		for j in range(dim):
			for l in range(n):
				s[j] += R[l, i] * (X[l, j] - mu[i, j])**2
			s[j] /= sum_Ri
		C[i, :, :] = np.diag(s)
	
	return prior, mu, C


def computeparameterssame(R, X):
	"""
	Computes the optimal parameters for the Gaussian mixture model with
	equal covariance matrices.
	
	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
    R : an array of size (n, k)
		responsibilities: R[i, j] = probability of data point i belonging to
		jth component.

	Returns
	-------
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		optimal means: mu[i, :] is the optimal mean of the ith component
	C : an array of size (k, m, m)
		optimal covariances: C[i, :, :] is the optimal covariance matrix of the
		ith components, the covariance matrices must be the same
	"""

	k = R.shape[1]
	n, dim = X.shape

	prior = np.zeros(k)
	mu = np.zeros((k, dim))
	C = np.zeros((k, dim, dim))
	sum_R = np.sum(R)

	# place your code here
	for i in range(k):
		sum_Ri = np.sum(R[:,i])
		prior[i] = sum_Ri/n
		mu[i,:] = np.sum(R[:,i].reshape((n,1))*X, axis=0) / sum_Ri
	
	for i in range(n):
		for j in range(k):
			C += R[i][j] * (X[i]-mu[j]).reshape(-1,1)@(X[i]-mu[j]).reshape(1,-1) / sum_R

	return prior, mu, C


def computeparametersspherical(R, X):
	"""
	Computes the optimal parameters for the Gaussian mixture model with
	equal diagonal spherical covariance matrices.
	
	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
    R : an array of size (n, k)
		responsibilities: R[i, j] = probability of data point i belonging to
		jth component.

	Returns
	-------
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		optimal means: mu[i, :] is the optimal mean of the ith component
	C : an array of size (k, m, m)
		Optimal covariances: C[i, :, :] is the optimal covariance diagonal
		matrix of the ith component. The numbers on the diagonals must be equal.
	"""
	k = R.shape[1]
	n, dim = X.shape

	prior = np.zeros(k)
	mu = np.zeros((k, dim))
	C = np.zeros((k, dim, dim))

	sum_R = np.sum(R)

	# place your code here
	for i in range(k):
		s = 0.
		sum_Ri = np.sum(R[:, i])
		prior[i] = sum_Ri/n
		mu[i, :] = np.sum(R[:, i].reshape((n,1))*X, axis=0) / sum_Ri
		for j in range(dim):
			for l in range(n):
				s += R[l, i] * (X[l, j] - mu[i, j])**2
		s /= dim
		C += s*np.eye(dim)/n
	
	return prior, mu, C


def computeresponsibilities(X, prior, mu, C):
	"""
	Computes responsibilities: R[i, j] = probability of data point i belonging
	to jth component.

	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		mu[i, :] is the mean of the ith component
	C : an array of size (k, m, m)
		C[i, :, :] is the covariance matrix of the ith component
	
	Returns
	-------
    R : an array of size (n, k)
		responsibilities: R[i, j] = probability of data point i belonging to
		jth component.
	"""

	def logGaussPDF(C, mu, x):
		m = C.shape[0]
		return -0.5*(np.log((2*np.pi)**m*np.linalg.det(C)) + (x-mu).reshape(-1,1).T @ np.linalg.inv(C) @ (x-mu))

	k = prior.shape[0]
	cnt = X.shape[0]

	# place your code here
	L = np.zeros((cnt, k))
	for i in range(cnt):
		for j in range(k):
			L[i, j] = np.log(prior[j]) + logGaussPDF(C[j, :, :], X[i, :], mu[j, :])
	norm = logsumrows(L).reshape((L.shape[0], 1))

	return np.exp(L - norm)


def em(X, R, itercnt, stats):
	"""
	EM algorithm: computes model parameters given the responsibilities and
	computes the responsibilities given the parameters.  Repeats itercnt times.

	Parameters
	----------
    X : an array of size (n, m)
		input data matrix
    R : an array of size (n, k)
		initial responsibilities: R[i, j] = probability of data point i
		belonging to jth component.
	itercnt : int
		number of iterations
	stats : function
		Function for computing the model parameters given the responsibilities,
		for example, computeparameters

	Returns
	-------
    R : an array of size (n, k)
		final responsibilities: R[i, j] = probability of data point i
		belonging to jth component.
	prior : an array of size k
		prior probabilities of the components
	mu : an array of size (k, m)
		mu[i, :] is the mean of the ith component
	C : an array of size (k, m, m)
		C[i, :, :] is the covariance matrix of the ith component

	"""

	# place your code here	
	for _ in range(itercnt):
		prior, mu, C = stats(R, X)
		R = computeresponsibilities(X, prior, mu, C)
	return R, prior, mu, C


def main(argv):
	np.random.seed(2022) # Forces random to be repeatable. Remove when random seed is desired. 

	X = np.loadtxt(argv[1])
	k = int(argv[2])
	mode = argv[3]
	itercnt = int(argv[4])

	n, m = X.shape
	R = np.random.rand(X.shape[0], k)
	R = R / R.sum(axis=1)[:,np.newaxis]
	print(R)

	if mode == 'normal':
		R, prior, mu, C = em(X, R, itercnt, computeparameters)
	elif mode == 'diag':
		R, prior, mu, C = em(X, R, itercnt, computeparametersdiagonal)
	elif mode == 'same':
		R, prior, mu, C = em(X, R, itercnt, computeparameterssame)
	elif mode == 'sphere':
		R, prior, mu, C = em(X, R, itercnt, computeparameterssame)
	else:
		print("Mode %s unrecognized" % mode)
		return

	print('R')
	print(R)
	print('priors')
	print(prior)
	print('means')
	print(mu)
	print('covariance matrices')
	print(C)



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__": 
	if len(sys.argv) != 5:
		print('usage: python %s filename number_of_factors iteration_count' % sys.argv[0])
	else:
		main(sys.argv)
