# PyMC #8306 — Distribution indexing type inference

**Change:** `pymc/distributions/distribution.py` now annotates `DistributionMeta.__call__` to return `TensorVariable`, delegating directly to `super().__call__(*args, **kwargs)`. This is the focused typing approach proposed in upstream PR [#8605](https://github.com/pymc-devs/pymc/pull/8605). It does not add a distribution-indexing shim or change how `Distribution.__new__` constructs the tensor.

**Regression coverage:** `tests/distributions/test_distribution.py` includes `test_issue_8306_distribution_constructor_indexing` for Normal and Poisson indexing and the metaclass return annotation. `ISSUE_8306_PYCHARM_REPRO.py` contains the original beta[0]/beta[1] reproduction, with runtime type assertions.

## Check in PyCharm

1. Open the extracted **`pymc-main`** directory as your project.
2. Select a Python interpreter where this source checkout is installed (PyCharm → Settings → Project → Python Interpreter). In that interpreter's **Terminal** run `python -m pip install -e .` if it is not already installed in editable mode. Avoid accidentally importing a different PyMC installation.
3. Open `ISSUE_8306_PYCHARM_REPRO.py`; inspect `beta[0]` and `beta[1]` for the original false-positive `__getitem__` warning.
4. Run `python ISSUE_8306_PYCHARM_REPRO.py` and `python -m pytest tests/distributions/test_distribution.py -k issue_8306 -q`.
5. If the old warning persists despite using the updated source, verify `pymc.__file__` in the project interpreter and refresh PyCharm indexes (File → Invalidate Caches).

**Validation boundary:** Static source/syntax checks were performed on the package. The patched checkout's full pytest suite and live PyCharm inspection were not possible in the packaging environment because its installed PyTensor is incompatible with this source snapshot. On Windows, previously seen PyTensor C-compiler failures in `test_issue_3051` are separate from the `beta[0]` static warning; this fix does not repair your C toolchain.

**Important:** The IDE-warning outcome still needs to be checked in your installed PyCharm version; the code change improves type inference but PyCharm behavior can vary by version.
