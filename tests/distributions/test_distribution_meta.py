import pymc as pm
from pymc.distributions.distribution import Distribution
from pytensor.tensor.variable import TensorVariable


def test_indexing_distribution_returns_tensor_variable():
    with pm.Model():
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        assert isinstance(beta, TensorVariable)
        assert isinstance(beta[0], TensorVariable)


def test_typing_shim_has_no_runtime_effect():
    # __getitem__ is declared only under TYPE_CHECKING (issue #8306)
    assert "__getitem__" not in vars(Distribution)
