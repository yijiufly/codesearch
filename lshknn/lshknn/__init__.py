import numpy as np
from ._lshknn import knn_from_signature
import pickle as p

class Lshknn:
    '''Local sensitive hashing k-nearest neighbors

    Calculate k-nearest neighbours of vectors in a high dimensional space
    using binary signatures of random hyperplanes as a coarse grain vector
    representation.

    Usage:

    >>> # Matrix with two features and three samples
    >>> data = np.array([[0, 1, 2], [3, 2, 1]])
    >>> knn, similarity, n_neighbors = Lshknn(data=data, k=1)()
    '''

    def __init__(
            self,
            data,
            k=20,
            threshold=0.4,
            m=100,
            slice_length=None,
            signature=None,
            set_index=0,
            query=0,
            ):
        '''Local sensitive hashing k-nearest neighbors.

        Arguments:
            data (dense or sparse matrix or dataframe): the \
vectors to analyze. Shape is (f, n) with f features and n samples.
            k (int): number of neighbors to find.
            threshold (float): minimal correlation threshold to be considered \
a neighbor.
            m (int): number of random hyperplanes to use.
            slice_length (int or None): The slice length to use if \
approximating signatures in hash buckets.

        Returns:
            (knn, similarity, n_neighbors): triple with the nearest \
neighbors, the matrix of similarities, and the number of neighbors for each \
cell.
        '''

        self.data = data
        self.signature = signature
        self.k = k
        self.threshold = threshold
        self.m = m
        #if self.data != None:
        self.n = data.size
        #else:
        #    self.n = signature.size
        self.slice_length = slice_length
        #self.signature = signature
        self.set_index = set_index
        self.query = query

    def _check_input(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('data should be a numpy 2D matrix')
        if len(self.data.shape) != 2:
            raise ValueError('data should be a 2D matrix')
        if np.min(self.data.shape) < 2:
            raise ValueError('data should be at least 2x2 in shape')
        if self.m < 1:
            raise ValueError('m should be 1 or more')
        if self.k < 1:
            raise ValueError('k should be 1 or more')
        if (self.threshold < -1) or (self.threshold > 1):
            raise ValueError('threshold should be between -1 and 1')
        if (self.slice_length is not None) and (self.slice_length < 0):
            raise ValueError('slice_length should be None or between 1 and 64')
        if (self.slice_length is not None) and (self.slice_length > self.m):
            raise ValueError('slice_length cannot be longer than m')
        if (self.set_index < 0) or (self.set_index > 5):
            raise ValueError('set index should between 0 and 5')

    def _normalize_input(self):
        if self.slice_length is None:
            self.slice_length = 0

        # Substract average across genes for each cell
        # FIXME: preserve sparsity?!
        self.data = self.data - self.data.mean(axis=0)

    def _generate_planes(self):
        # Optimization flags
        if self.data.flags['C_CONTIGUOUS']:
            planes = np.random.normal(
                    loc=0,
                    scale=1,
                    size=(self.m, self.data.shape[0]),
                    ).T
            planes /= planes.sum(axis=0)
            planes = planes.T
        else:
            planes = np.random.normal(
                    loc=0,
                    scale=1,
                    size=(self.data.shape[0], self.m),
                    )
            planes /= planes.sum(axis=0)
        f=open("/rhome/lgao027/bigdata/emb/knn_final/" + str(self.set_index) + "/random_plane/planes.p", "w")
        self.planes = planes
        p.dump(self.planes, f)
        f.close()

    def _compute_signature(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Generate planes first!')

        # NOTE: 90% of the time is spent here
        if self.data.flags['C_CONTIGUOUS']:
            signature = np.dot(self.planes, self.data).T > 0
            signature = np.ascontiguousarray(signature)
        else:
            signature = np.dot(self.data.T, self.planes) > 0
        
        #print signature
        word_count = 1 + (self.m - 1) // 64
        base = 1 << np.arange(64, dtype=np.uint64)
        ints = []
        for row in signature:
            for i in range(word_count):
                sig = row[i * 64: (i+1) * 64]
                ints.append(np.dot(sig, base[:len(sig)]))
        
        #print [ints]
        self.signature = np.array([ints], np.uint64)
        if self.query:
            f=open("/rhome/lgao027/bigdata/emb/knn_final/" + str(self.set_index) + "/random_plane/signature_query.p", "w")
        else:
            f=open("/rhome/lgao027/bigdata/emb/knn_final/" + str(self.set_index) + "/random_plane/signature.p", "w")
        p.dump(self.signature, f)
        f.close


    def _knnlsh(self):
            #raise AttributeError('Compute signature first!')

        # NOTE: I allocate the output array in Python for ownership purposes
        #self.knn = self.n * np.ones((self.n, self.k), dtype=np.uint64)
        #self.similarity = np.zeros((self.n, self.k), dtype=np.float64)
        #self.n_neighbors = np.zeros((self.n, 1), dtype=np.uint64)
        print("calling knnlsh")
        knn_from_signature(
                self.signature,
                #self.knn,
                #self.similarity,
                #self.n_neighbors,
                self.n,
                self.m,
                self.k,
                self.threshold,
                self.slice_length,
                self.set_index,
                self.query,
                )

    def _format_output(self):
        # Kill empty spots in the matrix
        # Note: this may take a while compared to the algorithm
        ind = self.knn >= self.n
        self.knn = np.ma.array(self.knn, mask=ind, copy=False)
        self.similarity = np.ma.array(self.similarity, mask=ind, copy=False)
        return self.knn, self.similarity, self.n_neighbors
    
    def _query(self):
        #self._normalize_input()
        f=open("/rhome/lgao027/bigdata/emb/knn_final/" + str(self.set_index) + "/random_plane/planes.p", "w")
        self.planes = p.load(f)
        f.close()
        self._compute_signature()
        f=open("/rhome/lgao027/bigdata/emb/knn_final/" + str(self.set_index) + "/random_plane/signature.p", "w")
        sig = p.load(f)
        f.close()
        sign = np.concatenate((sig[0], self.signature[0]))

        knn_from_signature(
                sign,
                #self.knn,
                #self.similarity,
                #self.n_neighbors,
                len(sign),
                self.m,
                self.k,
                self.threshold,
                self.slice_length,
                self.set_index,
                self.query,
        ) 
    def __call__(self):
        if self.query == 0:
            #self._check_input()
            #if self.signature == None:
            self._generate_planes()
            self._compute_signature()
        
            self._knnlsh()

        else:
            self._query()
