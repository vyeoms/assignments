import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances


def sammon(D, P, eta, tau, itercnt):
    """
    Sammon projection

    Parameters
    ----------
    D : an array of size (n, n)
        distance matrix between n data points, must be symmetric matrix
    P : an array of size (n, k)
        initial coordinates for the n data points in k-dimensional space
    eta : real
        parameter regulating the gradient step size (deterministic step)
    tau : real
        parameter regulating the gradient step size (deterministic step)
    itercnt : int
        number of iterations

    Returns
    -------
    P : an array of size (n, k)
        final coordinates for the n data points
    """

    n = D.shape[0]
    k = P.shape[1]
    C = np.zeros((n, k))

    c1 = -2 / np.sum(np.tril(D, -1))

    for i in range(itercnt):
        dij = euclidean_distances(P)

        for dim1 in range(n):
            for dim2 in range(k):
                c2 = 0.
                for j in range(n):
                    if j != dim1:
                        c2 += ((D[dim1,j] - dij[dim1,j]) / (dij[dim1,j] * D[dim1,j]))*\
                            (P[dim1,dim2] - P[j,dim2])
                C[dim1, dim2] = c1 * c2

        P -= eta*C/(1 + i*eta*tau)

    return P


def main(argv):
    X = np.loadtxt(argv[1])
    k = int(argv[2])
    eta = float(argv[3])
    tau = float(argv[4])
    itercnt = int(argv[5])

    D = euclidean_distances(X)

    pca = PCA(n_components=k)
    P = pca.fit_transform(X)
    print('PCA:')
    print(P)

    print('Sammon:')
    print(sammon(D, P, eta, tau, itercnt))


# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('usage: python %s filename proj_dim eta tau itercnt' % sys.argv[0])
    else:
        main(sys.argv)
