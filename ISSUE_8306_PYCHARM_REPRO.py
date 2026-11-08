"""Open this file in PyCharm to check issue #8306's indexing warning.

This is an IDE check: no warning should appear on beta[0] / beta[1].
Use a PyCharm interpreter with THIS repository installed in editable mode.
Run this file to check that the expressions work at runtime as well.
"""

import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable

size = 100
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

with pm.Model() as basic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    mu = alpha + beta[0] * X1 + beta[1] * X2

assert isinstance(beta, TensorVariable)
assert isinstance(beta[0], TensorVariable)
assert isinstance(beta[1], TensorVariable)
assert isinstance(mu, TensorVariable)
print("Issue #8306 runtime check passed; confirm in PyCharm that beta[0] has no warning.")
