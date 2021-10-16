"""
Module for categorical univariate Markov Model.
"""

from copy import deepcopy
from typing import List, Union
from scipy.sparse import csr_matrix
from scipy.optimize import LinearConstraint, Bounds, minimize
import numpy as np

from .base import ConditionalDistribution


class MarkovModel:
    """
    A Markov model for univariate categorical sequential data.
    """
    def __init__(self, K: int, M: int, random_init: bool = False):
        """
        Keyword arguments
        -----------------
        K: int -- number of categores for univariate data
        M: int -- order of Markov Model
        random_init: bool, default = False -- whether to instantiate the
            transition probabilities randomly. If False, transition states are
            instantiated uniformly.

        Examples
        --------
        >>> from pymm.models import MarkovModel
        >>>
        >>> model = MarkovModel(K=2, M=0)
        >>>
        >>> print(f'{model.M}th order Markov model with {model.K} categorical values')
        0th order Markov model with 2 categorical values
        """
        self.K = K
        self.M = M

        self.models = [ConditionalDistribution(K=K,
                                               M=_m,
                                               random_init=random_init
                                               )
                       for _m in range(M+1)]

        # total number of parameters
        self.N = sum([_m.N for _m in self.models])

    def log_prob(self, x: np.ndarray) -> float:
        """
        Returns ln p(x0) + ln p(x1|x0) ... + ln p(x_M-1|...x0) 
                + sum_n p(x_n | x_n-1...)

        Keyword arguments
        -----------------
        x: np.ndarray, shape = (N, ) -- sequence of univariate categorical
            values. Sequence is ordered left to right such that
            x = (x_0, x_1, x_2, ..., x_{N-1}).

        Examples
        --------
        >>> import numpy as np
        >>>
        >>> from pymm.models import MarkovModel
        >>>
        >>> model = MarkovModel(K=2, M=0)
        >>>
        >>> # ln p(x1=0, x2=1, x3=1)
        >>> model_result = model.log_prob(np.asarray([0, 1, 1]))
        >>>
        >>> expected_result = np.log(0.5)*3
        >>>
        >>> results_agree = np.isclose(model_result, expected_result)
        >>> print(f'Expected and numerical results agree? {results_agree}')
        Expected and numerical results agree? True

        Returns
        -------
        float -- the log joint probability ln p(x_0, x_1, ..., x_{N-1})
        """
        if x.shape[0] < self.M+1:
            raise ValueError("Input data must have length of "
                             + f" at least {self.M+1}")

        if self.M > 0:
            # ln p(x0) +...+ ln p(x_M-1 | ...x0)
            out = sum([self.models[_m].log_prob(*x[0:_m+1][::-1])
                       for _m in range(self.M)])
        else:
            out = 0.0

        for _n in range(self.M, x.shape[0]):
            out += self.models[self.M].log_prob(*x[_n-self.M:_n+1][::-1])

        return out

    def get_params(self) -> np.ndarray:
        """
        Returns a 1-d array of concatenated parameters over all self.models, in
        order of increasing model order.

        Examples
        --------
        > from pymm.models import MarkovModel
        >
        > model = MarkovModel(K=2, M=1)
        >
        > # Markov model of order 1 is composed of 2 constituent conditional
        > # distributions: p(x_n) and p(x_n | x_{n-1}).
        > # p(x_0, ..., x_N) = p(x_0) prod_{n=1}^N p(x_n | x_{n-1})
        > print(f'uniform initial probabilitiy states: {model.get_params()}')

        Returns
        -------
        np.ndarray, shape = (self.N, )
        """
        return np.concatenate(tuple(_m.get_params() for _m in self.models))

    def set_params(self, A_small: np.ndarray):
        """
        Sets _m.A for _m in self.models from a concatenated list of parameter
        values.

        Keyword arguments
        -----------------
        A_small: np.ndarray, shape = (self.N, ) -- all free parameters for all
            constituent conditional distribution of the Markov model.

        Examples
        --------
        > import numpy as np
        >
        > from pymm.models import MarkovModel
        > 
        > model = MarkovModel(K=3, M=0)
        >
        > # for 3 categorical values, there are 2 free parameters in a 0^th
        > # order Markov model
        > model.set_params(np.asarray([0.6, 0.2]))
        """
        # number of free params in each m^th order model
        N = [_m.N for _m in self.models]

        if sum(N) != A_small.shape[0]:
            raise Exception('Inconsistent shape of input argument.'
                            + f'Expected [{N}, ]')

        # start and finish slice indices for each model into A_small
        idx = [sum(N[0:i]) for i in range(self.M+1)]

        for mm in range(self.M):
            self.models[mm].set_params(A_small[idx[mm]: idx[mm+1]])
        self.models[-1].set_params(A_small[idx[-1]:])

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the jacobian of the log likelihood with respect to model free
        parameters (state transition probabilities). The jacobian of each m^th
        order model are concatenated into a 1-d array in increading model order.

        Keyword arguments
        -----------------
        x: np.ndarray, shape = [N, ] -- categorical data of length N and with
            self.K possible values = [0, self.K-1]
        """
        grad2 = [self.models[self.M].grad_log_prob(*x[_n-self.M:_n+1][::-1])
                 for _n in range(self.M, x.shape[0])]

        grad2 = np.sum(grad2, axis=0)

        if self.M > 0:
            grad1 = self.models[0].grad_log_prob(x[0])

            for _m in range(1, self.M):
                grads = self.models[_m].grad_log_prob(*x[0:_m+1][::-1])

                grad1 = np.concatenate((grad1, grads))

            return np.concatenate((grad1, grad2))
        else:
            return grad2

    def sample(self, N: int, x: Union[List, np.ndarray] = []) -> np.ndarray:
        """
        Return a sampled sequence of length N.

        Keyword arguments
        -----------------
        N: int -- The length of the sequence to return.
        x: Union[List, np.ndarray], default = [] -- An optional seed to begin
            sampling from.

        Examples
        ---------
        >>> import numpy as np
        >>> from pymm.models import MarkovModel
        >>>
        >>> model = MarkovModel(2, 1)
        >>> model.fit(np.asarray([0, 1, 0, 1, 0, 1]))
        >>>
        >>> # sequence depends on seed values
        >>> print(f'sequence 1: {model.sample(10, [0])}')
        sequence 1: [0 1 0 1 0 1 0 1 0 1]
        >>> print(f'sequence 2: {model.sample(10, [1])}')
        sequence 2: [1 0 1 0 1 0 1 0 1 0]

        """
        if N < 1:
            raise ValueError('Must sample at least 1 value.')

        x = list(deepcopy(x))

        if self.M > 0:
            for _m in range(self.M):
                if len(x) >= N:
                    break

                if len(x)-1 < _m:

                    # ConditionalDistribution.sample(*args) expects
                    # *args = [x_{n-1}, x_{n-2},...]
                    x.append(self.models[_m].sample(*x[:_m+1][::-1]))

        for _n in range(self.M, N):
            if len(x)-1 < _n:
                # ConditionalDistribution.sample(*args) expects
                # *args = [x_{n-1}, x_{n-2},...]
                x.append(self.models[self.M].sample(*x[_n-self.M:_n+1][::-1]))

        # shape = [N, ]
        return np.asarray(x)

    def _generate_constraint_matrix(self) -> csr_matrix:
        """
        Concatenate constraint matrix for all models into a single constraint
        matrix C such that 0 <= C * self.get_params() <= 1, which corresponds to
        the probability normalisation contraint.
        """
        # List[csr_matrix], len = self.M+1
        constraints = [_m.generate_constraint_matrix() for _m in self.models]

        if len(constraints) < 2:
            return constraints[0]
        else:
            out = self._stack(constraints[0], constraints[1])
            if len(constraints) < 3:
                return out
            else:
                for _c in constraints[2:]:
                    out = self._stack(out, _c)

                return out

    def _generate_constraint(self) -> LinearConstraint:
        """
        Returns a scipy.optimizse.LinearConstraint instance
        """
        C = self._generate_constraint_matrix()

        # avoid 1/0 in gradient of likelihood with respect to probabilities
        delta = 1e-6

        return LinearConstraint(C, np.zeros(C.shape[0])+delta,
                                np.ones(C.shape[0])-delta
                                )

    def _generate_bounds(self) -> Bounds:
        """
        Return a bounds object, constraining each transition probability
        component to be positive.

        Returns
        -------
        scipy.optimize.Bounds -- bounds to ensure that each transition state
            probability is positive and less then 1.
        """
        return Bounds(np.zeros(self.N), np.ones(self.N))

    def _log_prob(self, params: np.ndarray) -> float:
        """
        Returns log likelihood as an explicit function of free parameters,
        assuming self.x has been previously set.

        Keyword arguments
        -----------------
        params: np.ndarray, shape = (self.N, ) -- the Markov model transition
            state probabilities with which to evaluate the Markov model log
            likelihood of self.x.

        Returns
        -------
        float -- the log likelihood of the Markov model with supplied
            transition state probabilities and sequential data self.x
        """
        self.set_params(params)
        return -self.log_prob(self.x)

    def _grad_log_prob(self, params: np.ndarray) -> np.ndarray:
        """
        Returns gradient of log-likelihood with respect to transition state
        probability free parameters, assuming self.x: np.ndarray has been set.
        """
        self.set_params(params)
        return -self.grad_log_prob(self.x)

    def fit(self, x: np.ndarray):
        """
        Maximise log-joint likelihood with respect to transition state
        probabilities.

        Keyword arguments
        -----------------
        x: np.ndarray -- sequential data. Elements of x has must have values
            between [0, self.K-1] for K possible categorical values.

        Examples
        --------
        >>> import numpy as np
        >>>
        >>> from pymm.models import MarkovModel
        >>>
        >>> model = MarkovModel(K=3, M=1)
        >>>
        >>> model.fit(np.asarray([0, 1, 0, 0, 1, 0]))
        """
        self.x = x

        bounds = self._generate_bounds()
        constraints = self._generate_constraint()

        self.log = minimize(fun=self._log_prob,
                            jac=self._grad_log_prob,
                            x0=self.get_params(),
                            bounds=bounds,
                            constraints=constraints
                            )

    @classmethod
    def _stack(cls, matrix1: csr_matrix, matrix2: csr_matrix) -> csr_matrix:
        """
        Stack two matrices as block diagonal matrices.

        Keyword arguments
        -----------------
        matrix1: csr_matrix, shape = (nn1, dd1)
        matrix2: csr_matrix, shape = (nn2, dd2)

        Returns
        -------
        csr_matrix, shape = (nn1+nn2, dd1+dd2) -- the block diagonal matrix

            [matrix1, 0      ]
            [0,       matrix2]

            formed from stacking matrix1 and matrix2.
        """
        (nn1, dd1) = matrix1.shape
        (nn2, dd2) = matrix2.shape

        # shape = [n1+n2, d1+d2]
        C = csr_matrix((nn1+nn2, dd1+dd2))

        C[:nn1, :dd1] = matrix1[:, :]
        C[nn1:, dd1:] = matrix2[:, :]

        return C
