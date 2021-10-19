import numpy as np


from pymm.base import ConditionalDistribution


def test_constraint_length():
    """
    Check we get the correct shape for constraint matrix C:

    lower <= C x <= upper

    C: np.ndarray, shape = (m, n)
    (lower, upper): np.ndarray, shape = (m, )
    x: np.ndarray, shape = (n,)
    """

    for _k in np.arange(2, 6):
        for _m in np.arange(4):
            model = ConditionalDistribution(K=_k, M=_m)

            C = model.generate_constraint_matrix()

            # check shape
            assert C.shape == (int(model.N/(_k-1)),model.N)
