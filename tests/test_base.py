"""
Running pytest from the repository root will trigger evaluation
of all files

root/tests/*.py
"""
from copy import deepcopy
import numpy as np


from pymm.base import ConditionalDistribution


def _test_getset(K: int, M: int):
    inst = ConditionalDistribution(K=K, M=M, random_init=True)

    Aorig = deepcopy(inst.A)
    inst.set_params(inst.get_params())
    Arecon = deepcopy(inst.A)
    return np.allclose(Aorig, Arecon)


def test_getset():
    out = [[_test_getset(K=_k, M=_m) for _m in range(5)] for _k in range(2, 4)]
    assert np.all(out), 'mismatch between getter and setter'


def test_N():
    """
    test number of computed free parameters in transition state probability
    matrix.
    """
    for _k in np.arange(2, 6):
        for _m in np.arange(4):
            model = ConditionalDistribution(K=_k, M=_m)

            assert model.get_params().shape[0]==model.N
            assert model.A[:-1,...].flatten().shape[0]==model.N