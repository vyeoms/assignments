import sys
import numpy as np

def irls(X, labels, w, itercnt):
	"""
	IRLS algorithm

	Parameters
	----------
	X : an array of size (n, k)
		training input data for the classifier
	labels : an array of size n
		training labels for the classifier, elements must be 0/1
	w : an array of size k
		initial weights
	itercnt : int
		number of iterations

	Returns
	-------
	w : an array of size k
		weights after itercnt iterations
	err: an array of size itercnt + 1
		ith element correspongs to the error (objective function minimized in
		logistic regression) after the ith iteration. The 0th entry is the
		error with the initial weights.
	misclass: an array of size itercnt + 1
		ith element correspongs to the misclassification proportion after the
		ith iteration. The 0th entry is the misclassification proportion with
		the initial weights.
	"""

	err = np.zeros(itercnt + 1)
	misclass = np.zeros(itercnt + 1)

	y = 1 / (1 + np.exp(-X@w))
	err[0] = -np.sum(labels*np.log(y) + (1-labels)*np.log(1-y))
	misclass[0] = np.mean(labels != np.round(y))

	# Newton step itrcnt times
	for i in range(itercnt):
		S = np.diag(y*(1 - y))
		grad = X.T @ (y-labels)
		H = X.T @ S @ X
		
		# Newton update for w
		w = w - np.linalg.solve(H, grad)
		
		# Generate predictions
		y = 1 / (1 + np.exp(-X @ w))
		# Add a small epsilon for numerical stability
		err[i+1] = -np.sum(labels*np.log(y+1e-12) + (1-labels)*np.log(1-y+1e-12))
		misclass[i+1] = np.mean(labels != np.round(y))
	
	return w, err, misclass


def main(argv):
	D = np.loadtxt(argv[1])
	labels = D[:,0].copy() # copy is needed, otherwise next line will mess up the splice
	D[:,0] = 1 # replace the label column of D with constant, now the first feature gives us the bias term

	itercnt = int(argv[2])

	w = np.zeros(D.shape[1])
	w, err, misclass = irls(D, labels, w, itercnt)

	print('weights:')
	print(w)
	print('error:')
	print(err)
	print('misclassification rate:')
	print(misclass)



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__": 
	if len(sys.argv) != 3:
		print('usage: python %s filename' % sys.argv[0])
	else:
		main(sys.argv)
