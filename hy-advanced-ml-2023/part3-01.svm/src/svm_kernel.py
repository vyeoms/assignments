import sys
import numpy as np



class SVM:
	def __init__(self, C, kernel):
		""" 
		Initializes SVM with penalty C

		Parameters
		----------
		C : real
			penalty for violating the margin
		kernel : function
			kernel function, for example linear_kernel or rbf_kernel
		"""

		self.C = C
		self.kernel = kernel

	def step(self, i, j):
		""" 
		Optimizes alpha[i] and alpha[j] minimizing the dual program.
		Changes alpha[i], alpha[j]. Updates self.u

		Parameters
		----------
		i : int
			index referring to alpha[i]
		j : int
			index referring to alpha[j]

		Returns
		-------
		change : bool
			True if any alphas are updated
		"""

		alpha1 = self.alpha[i]
		alpha2 = self.alpha[j]
		y1 = self.y[i]
		y2 = self.y[j]
		E1 = self.u[i] - y1
		E2 = self.u[j] - y2

		# Find the bounds for alpha2
		if y1 != y2:
			L = max(0, alpha2 - alpha1)
			H = min(self.C, self.C + alpha2 - alpha1)
		else:
			L = max(0, alpha1 + alpha2 - self.C)
			H = min(self.C, alpha1 + alpha2)

		if L == H:
			return False

		# eta = self.X[i,:] @ self.X[i,:] + self.X[j,:] @ self.X[j,:] - 2 * self.X[i,:] @ self.X[j,:]
		eta = self.kernel(self.X[i,:], self.X[i,:]) + self.kernel(self.X[j,:], self.X[j,:]) - 2 * self.kernel(self.X[i,:], self.X[j,:])

		# minimize the dual objective by adding delta to alpha2 and -y1*y2*delta to alpha2
		# the dual program objective wrt. to delta becomes eta*delta^2 - 2*y2*(E1 - E2)*delta + constant

		if eta > 0: # objective w.r.t delta is quadratic -> find the new minimum point for alpha2 + delta
			n2 = alpha2 + y2*(E1 - E2) / eta
			n2 = min(H, max(L, n2))
		else: # objective w.r.t delta is linear -> check the boundaries and use the smallest
			if y2*(E1 - E2) < -10e-6:   # 10e-6 to deal with numerical instability
				n2 = L
			elif y2*(E1 - E2) > 10e-6:
				n2 = H
			else:
				return False
		

		if abs(n2 - alpha2) < 10e-6: # too small update, ignore
			return False


		n1 = alpha1 - y1*y2*(n2 - alpha2)
		n1 = min(self.C, max(0, n1)) # this is needed only for numerical stability 


		# update alphas
		self.alpha[i] = n1
		self.alpha[j] = n2

		
		# update predict (these can computed from scratch but it is faster to just update the difference
		n = self.X.shape[0]
		j_indices = np.zeros(n)
		i_indices = np.zeros(n)
		for z in range(n):
			j_indices[z] = self.kernel(self.X[z,:], self.X[j,:])
			i_indices[z] = self.kernel(self.X[z,:], self.X[i,:])
		
		self.u += (n2-alpha2) * j_indices * y2
		self.u += (n1-alpha1) * i_indices * y1


		return True


	def optimize(self):
		"""
		Fits SVM weights to the training data.
		Uses self.X and self.y as the training data.
		"""
		cnt = self.X.shape[0]
		
		# find optimal alphas
		changes = True
		round = 0
		giveup = 1000 # This is just so that the exercise doesn't run forever, if incorrect

		while changes and round < giveup:
			changes = False
			round += 1
			for i in range(cnt):
				for j in range(cnt):
					if self.step(i, j):
						changes = True

		# find b
		for i in range(cnt):
			if self.alpha[i] > 0 and self.alpha[i] < self.C:
				self.b = self.y[i] - self.u[i]
				break


	
	def fit(self, X, y):
		"""
		Fits SVM weights to the training data

		Parameters
		----------
		X : an array of size (n, k)
			input matrix
		y : an array of size n
			labels
		"""

		cnt = X.shape[0]
		self.X = X
		self.y = y
		self.u = np.zeros(cnt) # cached svm output for the training data, equal to selft.score(self.X)
		self.alpha = np.zeros(cnt)
		self.b = 0
		self.optimize()
	

	def score(self, X):
		"""
		Computes the score for input X, negative scores indicate label -1,
		positive scores indicate label 1.

		Parameters
		----------
		X : an array of size (n, k)
			input matrix
		Returns
		-------
		p : an array of size n
			scores, p[i] = sum_j alpha[j] y[j] <x[j], z> + bias,
			where
			x[j], y[j] is the jth training sample,
			z is the ith sample in X,
			and < , > is the inner product.
		"""
		n = self.X.shape[0]
		n1 = X.shape[0]
		k = np.zeros((n1, n))
		for i in range(n1):
			for j in range(n):
				k[i][j] = self.kernel(X[i], self.X[j])
		total = np.sum(k * self.alpha * self.y, axis=1)
		return total + self.b
		
	def predict(self, X):
		"""
		Predicts the labels for input X using self.score. Negative scores
		indicate label -1, positive scores indicate label 1.

		Parameters
		----------
		X : an array of size (n, k)
			input matrix

		Returns
		-------
		p : an array of size n
			predicted labels
		"""
		return np.sign(self.score(X))



def linear_kernel(x, y):
	return x @ y


def rbf_kernel(x, y):
	return np.exp(-0.5*np.sum((x - y)**2)) # RBF kernel with sigma = 1



def main(argv):
	D= np.loadtxt(argv[1])
	y = D[:,0]
	X = D[:,1:] 

	penalty = float(argv[2])

	svm = SVM(penalty, rbf_kernel)
	svm.fit(X, y)

	print('predictions for training data:')
	print(svm.predict(X))

	print('training error:')
	print(np.mean(svm.predict(X) != y))



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__": 
	if len(sys.argv) != 3:
		print('usage: python %s filename penalty' % sys.argv[0])
	else:
		main(sys.argv)
