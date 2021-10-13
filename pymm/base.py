"""
Base class for conditional state probabilities.
"""


import numpy as np
from scipy.sparse import csr_matrix


class ConditionalDistribution:
    """
    Conditional probability distribution p(x_n | x_n-1, .., x_n-M) for
    a M^th order Markox sequence.
    """
    def __init__(self, K: int, M: int, random_init: bool = False):
        """
        Keyword arguments
        -----------------
        K: int -- the number of possible values of the categorical variable.
            K can only take integer values greater then 1.
        M: int -- the order of the Markov model. M=0 corresponds to 
            p(x_n|x_{n-1},...x_1) = p(x_n). M cannot be less than 0.
        random_init: bool, default = False -- whether or not to instaniate
            transition probability matrix between states uniformly (when True)
            or randomly (when False). Useful for debugging and toy examples.

        Examples
        --------
        > from pymm.base import ConditionalDistribution
        >
        > # imagine data with 2 values of categorical variable
        > # create a 0th order Markov model - transition probabilities
        > # are independent of previous states.
        > model = ConditionalDistribution(K=2, M=0)
        """
        if K < 2:
            raise ValueError('Must have at least 2 possible '
                             + ' categorical value.')
        if M < 0:
            raise ValueError('Model order must be at least 0.')

        # number of possible categorical values
        self.K = K

        # order of Markovian dependence
        self.M = M

        self.init_A(random=random_init)

        # number of free parameters
        self.N = (self.K-1)*(self.K**self.M)

    def init_A(self, random: bool = False):
        """
        Instantiate and set state transition probabilities.

        Keyword arguments
        -----------------
        random: bool, default = False -- whether to instaniate
            state transition probabilities uniformly or randomly.

        Examples
        --------
        > from pymm.base import ConditionalDistribution
        >
        > # initiate weights uniformly
        > model = ConditionalDistribution(K=3, M=0, random_init=False)
        >
        > # oops, changed our mind - re-instantiate ranomdly
        > model.init_A(random=True)
        """
        if random:
            # N(0,1)
            self.A = np.random.normal(size=[self.K]*(self.M+1))

            # make positive
            self.A = np.exp(self.A)

            # normalize
            self.A /= np.tile(np.expand_dims(np.sum(self.A, axis=0), axis=0),
                              [self.A.shape[0]]+[1]*(len(self.A.shape)-1))
        else:
            # uniform initial values for transition probabilities
            self.A = np.ones([self.K]*(self.M+1))/self.K

    def prob(self, *args) -> float:
        """
        Return p(args[0]|args[1],args[2],..,args[M])

        args[0] = x_n
        args[1] = x_n-1
        .
        .
        .
        args[M] = x_n-M

        Assume that elements of *args are integers in the range 0 to K-1
        inclusive.

        Examples
        --------
        > from pymm.base import ConditionalDistribution
        >
        > 2nd order Markov model, 3 categorical values.
        > model = ConditionalDistribution(K=3, M=2)
        >
        > x1 = 0
        > x2 = 2
        > x3 = 1
        >
        > # p(x3|x2, x1)
        > p = model.prob(x3, x2, x1)
        """
        if len(args) != self.M+1:
            raise Exception('Input shape doesnt match expected shape')

        return self.A[tuple(args)]

    def log_prob(self, *args) -> float:
        return np.log(self.prob(*args))

    def grad_log_prob(self, *args) -> np.ndarray:
        """
        Returns the gradient of ln p(x_n|x_n-1..) with respect to A.

        A[-1,...] = 1-A[:-1,...] therefore

        grad A[:-1,...] = grad A[:-1,...] - grad A[-1,...]
        """
        out = np.zeros(self.A.shape)
        out[tuple(args)] = 1.0/self.A[tuple(args)]

        # shape = (K-1,...,K) -> (-1, )
        return np.reshape(out[:-1, ...] - out[-1, ...], (-1, ))

    def get_params(self) -> np.ndarray:
        """
        Returns 1-d array of free parameters.
        """
        # p(x_n=K-1 | x_n-1 = j) = 1 - sum_k=0^K-2 p(x_n=k | x_n-1 = j)
        return np.reshape(self.A[:-1, ...], (-1, ))

    def set_params(self, A_small: np.ndarray):
        """
        Set self.A, the (K,...,K) rank M+1 tensor for conditional
        transition probabilities from the (K-1, K, ..., K) rank M tensor
        A_small.

        Keyword arguments
        -----------------
        A_small, np.ndarray, shape = [(K-1)*K^M, ]

        Examples
        --------
        > from pymm.base import ConditionalDistribution
        >
        > model = ConditionalDistribution(K=2, M=1)
        >
        > # state transition probabilities
        > trans_probs = np.asarray([[0.2, 0.8],
                                    [0.3, 0.7]]
                                   )
        >
        > # 1-d array anticipated
        > model.set_params(trans_probs.flatten())
        """
        # rank M tensor, shape = (K-1, K, ...,K)
        A = np.reshape(A_small, tuple([self.K-1]+[self.K]*self.M))

        # rank M tensor, shape = (1, K, ..., K)
        A_ = 1.0 - np.reshape(np.sum(A, axis=0), tuple([1]+[self.K]*self.M))

        if self.M > 0:
            # rank M tensor, shape = (K,...,K)
            self.A = np.vstack((A, A_))
        else:
            self.A = np.hstack((A, A_))

    def sample(self, *args) -> int:
        """
        Return a sample conditioned on the Markov blanket, *args.

        For example, for a model of order self.M=2, this class models
        the transition states p(x3|x2,x1). *args=[x2, x1] will generate
        a sample x3 ~ p(x3|x2,x1). Note the assumed order of x2 and x1
        in *args.

        Keyword arguments
        -----------------
        *args: List[float] -- for *args=[x2,x1] (self.M=2), return x3
            such that x3 ~ p(x3|x2,x1).
        """
        if len(args) != self.M:
            raise ValueError('Arguments must be a list of values of '
                             + 'Markov blanket.')

        # transition probabilities to state k, shape = (K, )
        probs = self.A.T[tuple(args[::-1])]

        return np.where(np.random.multinomial(1, pvals=probs))[0][0]

    def generate_constraint_matrix(self) -> csr_matrix:
        """
        Generate a matrix C such that 0 <= C self.get_params() <= 1
        define the constraints that sum_{x_n} p(x_n | x_n-1,...) = 1

        C.shape = ((K-1)^M, N)
        """
        C = np.zeros((self.K**self.M, self.N))

        for cc in range(C.shape[0]):
            C[cc, (self.K-1)*cc:(self.K-1)*(cc+1)] = 1.0

        return csr_matrix(C)
