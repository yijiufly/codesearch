import numpy as np
from scipy.sparse import csr_matrix


def read_graph(path):
    '''
    Read the signed network from the path

    inputs
        path: string
            path for input file

    outputs
        A: csr_matrix
            signed adjacency matrix
        base: int
            base number of node id (start from base)
    '''

    X = np.loadtxt(path, dtype=float, comments='#')
    m, n = X.shape

    if n <= 2 or n >= 4:
        raise FormatError('Invalid input format')

    base = np.amin(X[:, 0:2])

    if base < 0:
        raise ValueError('Out of range of node id: negative base')

    X[:, 0:2] = X[:, 0:2] - base
    rows = np.concatenate((X[:, 0], X[:, 1]), axis = 0)
    cols = np.concatenate((X[:, 1], X[:, 0]), axis = 0)
    data = np.concatenate((X[:, 2], X[:, 2]), axis = 0)

    n = int(np.amax(X[:, 0:2]) + 1)

    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    return A, int(base)
