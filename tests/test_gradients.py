from copy import deepcopy
import numpy as np
from scipy.optimize import approx_fprime


from pymm.base import ConditionalDistribution
from pymm.models import MarkovModel


def _test_grad(model, x: np.ndarray):
    def log_prob(params: np.ndarray):
        # log prob as function of free model params
        # for fixed sequence x
        _model = deepcopy(model)
        _model.set_params(params)
        if isinstance(_model, ConditionalDistribution):
            return _model.log_prob(*x)
        else:
            return _model.log_prob(x)

    numerical_grad = approx_fprime(model.get_params(), log_prob, 1e-8)
    if isinstance(model, ConditionalDistribution):
        analytical_grad = model.grad_log_prob(*x)
    else:
        analytical_grad = model.grad_log_prob(x)

    assert np.allclose(numerical_grad, analytical_grad),\
        f'gradient computation wrong: {analytical_grad} != {numerical_grad}'


def test_grad_ConditionalDistribution():
    # sequence
    x = np.asarray([1, 1], dtype=int)

    model = ConditionalDistribution(K=2, M=1, random_init=True)

    _test_grad(model, x)


def test_grad_MarkovModel():
    """
    Comparae finite and analytical derivative of log likelihood with
    respect to all model parameters.
    """
    # sequence
    x = np.asarray([1, 1, 0, 1, 0], dtype=int)

    for _m in range(4):
        model = MarkovModel(K=2, M=1, random_init=True)

        _test_grad(model, x)
