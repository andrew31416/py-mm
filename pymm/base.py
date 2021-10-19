"""
Base class for conditional state probabilities.
"""


import numpy as np
from scipy.sparse import csr_matrix


class ConditionalDistribution:
    """
    Base class to represent conditional probability distributions of single
    order. For M=2, we have p(x_{n}|x_{n-1},...,x_0) = p(x_{n}|x_{n-1},x_{n-2}).
    This class assumes that we always provide a conditional sequence of
    necessary length. We cannot for example evaluate p(x_{n}|x_{n-1},x_{n-2})
    for an order M=2 model when we only have (x_n, x_{n-1}). For this case, we
    need to use pymm.models.MarkovModel, which composes sequences of increasing
    order pymm.base.ConditionalDistribution class instances.
    """
    def __init__(self, K: int, M: int, random_init: bool = False):
        """
        Keyword arguments
        -----------------
        K: int -- the number of possible values of the categorical variable. K
            can only take integer values greater then 1.
        M: int -- the order of the Markov model. M=0 corresponds to
            p(x_n|x_{n-1},...x_1) = p(x_n). M cannot be less than 0.
        random_init: bool, default = False -- whether or not to instaniate
            transition probability matrix between states uniformly (when True)
            or randomly (when False). Useful for debugging and toy examples.

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> # imagine data with 2 values of categorical variable
        >>> # create a 0th order Markov model - transition probabilities
        >>> # are independent of previous states.
        >>> model = ConditionalDistribution(K=2, M=0)
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
        Instantiate and set initial state transition probabilities.

        Keyword arguments
        -----------------
        random: bool, default = False -- whether to instaniate
            state transition probabilities uniformly or randomly.

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> # initiate weights uniformly
        >>> model = ConditionalDistribution(K=3, M=0, random_init=False)
        >>>
        >>> # oops, changed our mind - re-instantiate ranomdly
        >>> model.init_A(random=True)
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
        Returns the conditional probability distribution
        p(args[0]|args[1],args[2],..,args[M])

        Keyword arguments
        -----------------
        *args = List[int] -- the sequence of categorical variables in reverse
            order. args = [x_{n}, x_{n-1},...,x_{n-M}]. Elements of *args must
            be integers in range 0 to self.K-1, inclusive.

        Returns
        -------
        float -- the conditional probability distribution p(args[0]|*args[1:]).

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> # 2nd order Markov model, 3 categorical values.
        >>> model = ConditionalDistribution(K=3, M=2)
        >>>
        >>> x1 = 0
        >>> x2 = 2
        >>> x3 = 1
        >>>
        >>> # p(x3|x2, x1)
        >>> p = model.prob(x3, x2, x1)
        """
        if len(args) != self.M+1:
            raise Exception('Input shape doesnt match expected shape')

        return self.A[tuple(args)]

    def log_prob(self, *args) -> float:
        """
        Returns the log conditional probability distribution
        p(x_{n}|x_{n-1},...,x_{n-M}) for an order M conditional distribution.

        Keyword arguments
        -----------------
        *args = List[int] -- the sequence of categorical variables in reverse
            order. args = [x_{n}, x_{n-1},...,x_{n-M}]. Elements of *args must
            be integers in range 0 to self.K-1, inclusive.

        Returns
        -------
        float -- the log conditional probability distribution
            ln p (x_{n}|x_{n-1},...,x_{n-M}), where args=[x_{n},...,x_{n-M}].

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> # instantiate uniform transition state probabilities for an order 3
        >>> # conditional distribution
        >>> model = ConditionalDistribution(K=3, M=3)
        >>>
        >>> x4 = 2
        >>> x3 = 0
        >>> x2 = 1
        >>> x1 = 0
        >>>
        >>> # p(x4 | x3, x2, x1) for the sequence [x1, x2, x3, x4]
        >>> log_prob = model.log_prob(x4, x3, x2, x1)
        """
        return np.log(self.prob(*args))

    def grad_log_prob(self, *args) -> np.ndarray:
        """
        Returns the gradient of ln p(x_n|x_{n-1},...,x_{n-M}) for an order M
        conditional distribution, with respect to transition state probabilities
        self.A.

        Because of the constraint that

        sum_{i=0}^{K-1} p(x_n=i | x_{n-1},...,x_{n-M}) = 1, we can write that

        p(x_n = K-1 | ...) = 1-sum_{i=0}^{K-2} p(x_n=i | ...) and therefore

        A[-1,...] = 1-sum(A[:-1,...], axis=0) therefore

        grad A[:-1,...] = grad A[:-1,...] - grad A[-1,...]


        Keyword arguments
        -----------------
        args: List[int] -- the categorical sequence in reverse order.
            args = [x_{n}, x_{n-1},..., x_{n-M}].

        Returns
        -------
        np.ndarray, shape = (K-1,...,K), rank = M+1 -- the jacobian of the
            log conditional distribution ln p(x_n|x_{n-1},...,x_{n-M}), with
            respect to the free transition state probabilities. The fact that we
            have normalization constraints reduces the length of the first
            dimension to be 1 less than self.A, the transition state
            probabilities.

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> model = ConditionalDistribution(K=2, M=0)
        >>>
        >>> # grad p(x=1)
        >>> jac = model.grad_log_prob(1)

        """
        out = np.zeros(self.A.shape)
        out[tuple(args)] = 1.0/self.A[tuple(args)]

        # shape = (K-1,...,K) -> (-1, )
        return np.reshape(out[:-1, ...] - out[-1, ...], (-1, ))

    def get_params(self) -> np.ndarray:
        """
        Returns 1-d array of free parameters. This is smaller in size then the
        transition state probabilies, because we account for the normalization
        constraints.

        Returns
        -------
        np.ndarray, shape = (self.N, ) -- the free parameters of transition
            state probabilities.
        """
        # p(x_n=K-1 | x_n-1 = j) = 1 - sum_k=0^K-2 p(x_n=k | x_n-1 = j)
        return np.reshape(self.A[:-1, ...], (-1, ), order='F')

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
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> model = ConditionalDistribution(K=2, M=1)
        >>>
        >>> # state transition probabilities
        >>> trans_probs = np.asarray([[0.2, 0.8],[0.3, 0.7]])
        >>>
        >>> # 1-d array anticipated. We only set free parameters and ignore
        >>> # final element of first-axis in trans_probs, since these are
        >>> # known through the normalization constraint
        >>> model.set_params(trans_probs[0, :].flatten())
        """
        # rank M tensor, shape = (K-1, K, ...,K)
        A = np.reshape(A_small, tuple([self.K-1]+[self.K]*self.M), order='F')

        # rank M tensor, shape = (1, K, ..., K)
        A_ = 1.0 - np.reshape(np.sum(A, axis=0), tuple([1]+[self.K]*self.M))

        if self.M > 0:
            # rank M tensor, shape = (K,...,K)
            self.A = np.vstack((A, A_))
        else:
            self.A = np.hstack((A, A_))

    def sample(self, *args) -> int:
        """
        Return a sample x_n from the conditional distribution
        p(x_n | x_{n-1},...,x_{n-M}).

        Keyword arguments
        -----------------
        *args: List[int] -- the sequence that x_n is conditioned on.
            args=[x_{n-1},..., x_{n-M}] is the sequence [x_{n-M},...,x_{n-1}] in
            reverse order.

        Returns
        -------
        int = [0, self.K-1] -- returns a sample x_n from the conditional
            distribtion p(x_n | x_{n-1},...,x_{n-M}), where
            args = [x_{n-1},...,x_{n-M}] represents the sequence
            [x_{n-m},...x_{n-1}] in reverse order, that the distribution for x_n
            is conditioned on. The sampled value is an integer in the range
            [0, self.K-1], inclusive.

        Examples
        --------
        >>> from pymm.base import ConditionalDistribution
        >>>
        >>> # order 0 model with 4 categorical values
        >>> model0 = ConditionalDistribution(K=4, M=0)
        >>>
        >>> # samples from an order 0 conditional distribution are independent
        >>> # of all preceeding values in the sequence.
        >>> x = model0.sample()
        >>>
        >>> # order 2 model with 2 categorical values
        >>> model2 = ConditionalDistribution(K=2, M=2)
        >>>
        >>> x1 = 1
        >>> x2 = 0
        >>> # x3 ~ p(x3 | x2, x1)
        >>> x3 = model2.sample(x2, x1)
        >>>
        >>> # generated sequence from order 2 model. First 2 elements were seed
        >>> # values
        >>> sequence = [x1, x2, x3]
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
