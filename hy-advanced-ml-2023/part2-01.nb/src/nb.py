import sys
import numpy as np


def nb(X, labels):
	"""
	Computes the weight vector w and and bias b corresponding
	to the Naive Bayes classifier with Bernoulli components.

	Parameters
	----------
	X : an array of size (n, k)
		training input data for the classifier, elements must be 0/1
	labels : an array of size n
		training labels for the classifier, elements must be 0/1

	Returns
	-------
	w : an array of size k
		weights corresponding to the classifier
	bias: real number
		bias term corresponding to the classifier
	"""

	cnt = X.shape[0]

	prior_y1 = np.sum(labels) / cnt
	
	# get the conditional probabilities of each feature (columns): p(x=1|y)
	cond_y0 = np.mean(X[labels == 0], axis=0)
	cond_y1 = np.mean(X[labels == 1], axis=0)
	
	# get the weights and bias: remember that log(x/a) = log(x) - log(a); log(xy) = log(x) + log(y)
	w = np.log(cond_y1)+np.log(1-cond_y0) - np.log(1-cond_y1) - np.log(cond_y0)
	b = np.sum(np.log(1-cond_y1) - np.log(1-cond_y0)) + np.log(prior_y1) - np.log(1-prior_y1)
	
	return w, b

def main(argv):
	D = np.loadtxt(argv[1])
	X = D[:, 1:]
	labels = D[:,0]
	print(nb(X, labels))



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__": 
	if len(sys.argv) != 2:
		print('usage: python %s filename' % sys.argv[0])
	else:
		main(sys.argv)
