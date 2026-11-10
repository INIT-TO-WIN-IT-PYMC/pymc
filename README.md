# MachZero — Team 2 | PyMC Issue #8306

**Topic:** Incorrect IDE type warning when indexing PyMC distributions  
**Project:** [PyMC](https://github.com/pymc-devs/pymc) · **Issue:** [#8306](https://github.com/pymc-devs/pymc/issues/8306) · **Reference PR:** [#8605](https://github.com/pymc-devs/pymc/pull/8605)

This README follows our work in three rounds: **understanding the problem, deciding on an approach, and implementing and testing the change**.

---

## Round 1 — Problem Statement and Issue Identification

### What was the problem?

When a developer creates a PyMC distribution, the result is a PyTensor `TensorVariable`. We can index or slice that variable just like other tensors. However, an IDE such as PyCharm can incorrectly infer that the result is a `Normal` class instance, and then show an error on valid indexing code.

**Example that triggers the IDE warning:**

```python
import pymc as pm

with pm.Model():
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    mu = beta[0] + beta[1]  # PyCharm may flag this indexing
```

**Warning displayed in PyCharm:**

> Class 'Normal' does not define '__getitem__', so the '[]' operator cannot be used on its instances.

![PyCharm warning on beta indexing](assets/pycharm-indexing-warning.png)

*Figure 1 — The reported false-positive warning on `beta[0]` / `beta[1]`.*

### Why does it happen?

- **At runtime:** `pm.Normal(...)` creates a `TensorVariable`; indexing `beta[0]` works.
- **During IDE analysis:** the editor may infer `Normal`, which does not itself expose tensor indexing in the way the IDE expects.
- **The mismatch:** the IDE reports an error even though Python executes the expression successfully.

**Our goal:** help IDEs recognize the tensor return type **without changing PyMC's distribution behavior, calculations, or sampling logic**.

---

## Round 2 — Understanding and Proposed Approach

### How we thought about the issue

We first checked whether the warning represented a real Python error. The indexing example ran, so we focused on **type information presented to the IDE**, rather than on rewriting the indexing operation.

We then looked for a common point shared by distributions. Updating every individual distribution (`Normal`, `Poisson`, and others) would be repetitive and harder to maintain. A centralized typing improvement seemed cleaner.

### Relevant codebase structure

```text
pymc-main/
├── pymc/
│   └── distributions/
│       ├── distribution.py        # DistributionMeta and shared construction logic
│       ├── continuous.py          # Normal and other continuous distributions
│       ├── discrete.py            # Poisson and other discrete distributions
│       └── multivariate.py        # Multivariate distribution implementations
├── tests/
│   └── distributions/
│       └── test_distribution.py   # Existing tests and added regression test
├── ISSUE_8306_PYCHARM_REPRO.py    # Runtime example and IDE reproduction
└── pyproject.toml                 # Package and test configuration
```

### Idea and intended workflow (before implementation)

```text
User calls pm.Normal("beta", shape=2)
                   |
                   v
       Shared distribution construction
           (DistributionMeta)
                   |
         +---------+----------+
         |                    |
         v                    v
   IDE type analysis     Python execution
   should see a          creates a real
   TensorVariable        TensorVariable
         |                    |
         +---------+----------+
                   v
        beta[0] and beta[1] are valid
```

**Proposed direction:** add a return-type annotation at the metaclass call boundary, so an IDE has a clearer indication of what the distribution constructor returns. The approach should preserve existing execution by forwarding calls to the original constructor.

**Why this approach?** It keeps the change small, applies in one shared location, and avoids suppressing warnings individually in user code. Whether a specific PyCharm version actually respects the annotation still needs to be checked in the editor.

---

## Round 3 — Implementation, Process, and Testing

### 1. What we changed

**Primary file:** `pymc/distributions/distribution.py`

**Before (original source):** `DistributionMeta` defined `__new__` but did not explicitly declare a tensor return type for `__call__`.

```python
class DistributionMeta(ABCMeta):
    # The original class did not declare __call__ here.
    def __new__(cls, name, bases, clsdict):
        ...  # Existing implementation omitted for brevity
```

**After (our typing change):**

```python
class DistributionMeta(ABCMeta):
    def __call__(cls, *args: Any, **kwargs: Any) -> TensorVariable:
        """Instantiate a distribution, which returns a tensor variable."""
        return super().__call__(*args, **kwargs)
```

The added method advertises a `TensorVariable` return type. It still delegates construction to `super().__call__`, so the existing tensor creation and sampling code is not replaced. `ABCMeta`, `Any`, and `TensorVariable` are already imported in the module.

**After (the intended user-facing behavior):**

```python
import pymc as pm
from pytensor.tensor.variable import TensorVariable

with pm.Model():
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    mu = beta[0] + beta[1]

assert isinstance(beta, TensorVariable)
assert isinstance(mu, TensorVariable)
```

The user code does **not** need to be rewritten. The aim of the change is to improve how the IDE understands it.

### 2. Steps we followed

1. **Reproduced the warning** using `beta[0]` and `beta[1]` in PyCharm.
2. **Checked the runtime behavior** and confirmed that indexing works when executed.
3. **Traced distribution construction** through `distribution.py` and reviewed related distribution files.
4. **Added a central return annotation** to `DistributionMeta.__call__` instead of patching each distribution.
5. **Added regression coverage** for the annotation and indexability of `Normal` and `Poisson` variables.
6. **Ran the reproduction and distribution tests**; separated an unrelated Windows compiler error from the IDE type-inference issue.

### 3. Test cases

The added test, `test_issue_8306_distribution_constructor_indexing`, checks the metaclass annotation and verifies that indexing `Normal` and `Poisson` results produces `TensorVariable` objects.

| Test / check | Expected behavior | Recorded result |
|---|---|---|
| `ISSUE_8306_PYCHARM_REPRO.py` | `beta[0]` / `beta[1]` execute and produce tensors | **Passed** |
| New `test_issue_8306_distribution_constructor_indexing` | Correct return annotation; `Normal` and `Poisson` remain indexable | **Included in the passing distribution-test run** |
| Existing multivariate tests (`test_issue_3051`) | Multivariate log-probability calculations work | **Passed with C compilation disabled** |
| Full `tests/distributions/test_distribution.py` with `cxx=` | Check for distribution regressions | **30 passed, 2 XPASS(strict), 5 warnings** |
| PyCharm visual inspection | No false-positive `__getitem__` warning on `beta[0]` | **Not resolved/verified: supplied screenshot still shows the warning** |

**Runtime verification command:**

```powershell
python ISSUE_8306_PYCHARM_REPRO.py
```

**Recorded output:**

```text
Issue #8306 runtime check passed; confirm in PyCharm that beta[0] has no warning.
```

![PyCharm terminal showing passing runtime check](assets/runtime-check-passed.png)

*Figure 2 — Runtime reproduction succeeded. This is not proof that PyCharm's warning disappeared.*

**Distribution tests (PyCharm PowerShell terminal):**

```powershell
$env:PYTENSOR_FLAGS="cxx="
python -m pytest tests\distributions\test_distribution.py -q
```

**Recorded summary:**

```text
2 failed, 30 passed, 5 warnings in 5.98s
```

The two reported failures were `XPASS(strict)` for `TestDiracDelta.test_dtype[float32]` and `[float64]`: those Windows tests are marked *expected to fail* but their assertions actually passed. Because the project enables strict expected-failure handling, pytest counts those unexpected passes as failures. These are separate from issue #8306.

**Optional follow-up to run those tests without expected-failure handling (not yet recorded):**

```powershell
python -m pytest tests\distributions\test_distribution.py -q --runxfail
```

### 4. Challenges and current status

- **Compiler problem:** the original tests encountered `C:\MinGW\bin\g++.EXE: sorry, unimplemented: 64-bit mode not compiled in`. The installed compiler could not build 64-bit modules. Using `PYTENSOR_FLAGS="cxx="` bypassed that toolchain for the diagnostic run.
- **Unexpected passes:** the two Windows-specific `XPASS(strict)` cases above are test-marker issues, not broken indexing.
- **IDE warning persists in the supplied screenshot:** although the runtime test passes and the typing change is in place, we cannot claim the PyCharm warning has been eliminated. The next check is to confirm that PyCharm indexes this editable checkout, refresh its caches if needed, and evaluate whether the metaclass annotation is recognized by that IDE version.

**Status:** The central typing change and runtime tests are complete. **Final resolution of the PyCharm static warning remains unconfirmed.**

---

## Development Environment and System Requirements

### Environment used

| Component | Observed version / setup |
|---|---|
| Operating system | **Windows 11, 64-bit**, build `10.0.26200` |
| IDE | **PyCharm** (exact version not recorded) |
| Python | **3.14.0, 64-bit** |
| Terminal | **PowerShell** inside PyCharm |
| Virtual environment | **`.venv`** (project-local) |
| Package management | **pip**, with PyMC installed using `pip install -e .` |
| Test framework | **pytest 9.2.0** (shown in the installation screenshot) |
| PyMC / PyTensor / NumPy | Installed from the project requirements; exact installed versions not recorded |
| C++ compiler initially used | `C:\MinGW\bin\g++.EXE` (not capable of 64-bit compilation on this setup) |
| Diagnostic configuration | `PYTENSOR_FLAGS="cxx="` (disables PyTensor C++ compilation) |
| Project location | `D:\mechzero\PyMC_Issue_8306_Fixed\pymc-main` |

The project's `pyproject.toml` allows **Python `>=3.12,<3.15`**. The environment documented above uses **3.14.0**; other versions were not tested in this local run.

### What you need installed

- **64-bit Windows and a supported 64-bit Python version** (Python 3.14.0 was used here).
- **PyCharm** to reproduce and inspect the IDE warning; set the project interpreter to `.venv\Scripts\python.exe`.
- **Python `venv` and `pip`** for an isolated environment and dependency installation.
- **The extracted PyMC project**, opened at the folder containing `pyproject.toml`.
- **Internet access during initial installation** if package dependencies are not already cached; **Git** is optional when starting from the ZIP.
- **A compatible 64-bit C++ compiler** for native C++ compilation tests. It is *not required* for the diagnostic `cxx=` mode or the IDE-warning reproduction.

### Setup and reproduction commands

Run these from the **PyCharm PowerShell terminal**, inside the extracted `pymc-main` folder:

```powershell
cd D:\mechzero\PyMC_Issue_8306_Fixed\pymc-main
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest

# Check imported source and installed versions
python -c "import pymc; print(pymc.__file__)"
python --version
python -m pip show pymc pytensor numpy pytest

# Runtime reproduction
python ISSUE_8306_PYCHARM_REPRO.py

# Tests with the incompatible C++ compiler disabled
$env:PYTENSOR_FLAGS="cxx="
python -m pytest tests\distributions\test_distribution.py -q
```

If an appropriate `.venv` already exists, **skip creating it again** and simply activate it.

### Final takeaway

Issue #8306 is about **incorrect IDE type inference, not invalid tensor indexing**. We implemented a small, centralized type-annotation change and verified that indexing still executes. The remaining work is to demonstrate that the targeted PyCharm version actually stops showing the warning; the supplied IDE screenshot indicates that this has **not yet been established**.
