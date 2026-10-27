# glmstar

A Python implementation of the glmnet algorithm for fitting generalized linear models via penalized maximum likelihood. This package provides fast and efficient regularization paths for linear, logistic, multinomial, Poisson, and Cox regression models.

## Features

- **Multiple GLM Families**: Support for Gaussian (linear), Binomial (logistic), Multinomial, Poisson, and Cox regression
- **Elastic Net Regularization**: Combines L1 (Lasso) and L2 (Ridge) penalties with mixing parameter α
- **Fast Path Algorithms**: Efficient computation of regularization paths using coordinate descent
- **Cross-Validation**: Built-in cross-validation for optimal λ selection
- **Prediction Methods**: Interpolation and prediction at arbitrary λ values
- **sklearn Compatibility**: Follows scikit-learn API conventions
- **C++ Backend**: High-performance C++ implementation with pybind11 bindings

## Installation

### From PyPI

```bash
pip install glmstar
```

Wheels are on PyPI for Python 3.11 to 3.14 (glmstar 0.6.0 also for 3.10) on Linux
(x86-64), macOS (Apple silicon) and Windows; elsewhere pip builds from the source
distribution, which includes glmnetpp and Eigen and needs a C++17 compiler. The package
imports as `glmnet`.

### Get the source

The C++ code depends on two git submodules, [Eigen](https://gitlab.com/libeigen/eigen)
(from its GitHub mirror) and [glmnetpp](https://github.com/intro-stat-learning/glmnetpp),
so clone recursively. glmnetpp is a private repository, cloned over SSH: cloning it needs
access to it and an SSH key on GitHub (without them, install from PyPI, whose source
distribution includes it).

```bash
git clone --recursive https://github.com/jonathan-taylor/glmstar.git
cd glmstar

# Or, if already cloned without submodules
git submodule update --init --recursive
```

A C++17 compiler is required to build the extension modules.

### Development install with uv (recommended)

```bash
uv venv                         # .venv in the project
source .venv/bin/activate       # (uv finds .venv without it; python, pytest need it)
uv pip install --group build    # the build tools first (meson-python, coxdev, ...)
uv sync                         # editable install plus the test tools (the `dev` group)
uv run pytest tests --test-size=small
```

The project uses meson-python, whose editable installs recompile the C++ extensions on
import. For that reason `pyproject.toml` disables build isolation for `glmstar` under
uv, so the build tools must live in the project environment, and they have to be there
before uv first builds glmstar: a plain `uv sync` in a new environment fails with
`No module named 'mesonpy'`. To target an already-activated venv other than `.venv`,
use `uv sync --active`.

If you prefer `uv pip`, install the build tools first in the same way:

```bash
uv pip install --group build
uv pip install -e .      # or `uv pip install .` for a regular install
```

### Install with pip

In a virtual environment of its own:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install .
```

For an editable install with pip, again install the build tools into the environment
and disable build isolation:

```bash
pip install meson-python meson ninja pybind11 setuptools_scm coxdev
pip install --no-build-isolation -e .
```

### Building the docs

The documentation is a Jupyter Book 2 site, its pages executed, and a Sphinx API
reference. It needs [Node.js](https://nodejs.org) 18 or later (Jupyter Book 2 runs on
it), but not R. In the activated environment of the development install above, from the
repository's root:

```bash
uv sync --extra doc --extra gam     # Jupyter Book 2, Sphinx, lassoinf, jax
uv pip install --no-deps ISLP       # the data of some pages, without ISLP's dependencies
cd docs
jupyter-book build --html --execute         # the pages, executed: docs/_build/html
sphinx-build -b html api _build/html/api    # the API reference: docs/_build/html/api
open _build/html/index.html
```

`jupyter-book start` serves the pages instead, rebuilding them as they change. This is
what `.github/workflows/build_docs.yml` runs.

### Troubleshooting

If the build fails with `'pybind11/pybind11.h' file not found`, a stale meson build
directory (e.g. `build/cp312`) is pointing at a temporary build environment that no
longer exists. Delete it and reinstall:

```bash
rm -rf build/cp*
uv sync
```

## Quick Start

```python
import numpy as np
from glmnet import GaussNet, LogNet
from sklearn.datasets import make_regression, make_classification

# Linear Regression
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, random_state=42)
fit_gaussian = GaussNet().fit(X, y)

# Plot coefficient paths
ax = fit_gaussian.coef_path_.plot()
ax.set_title('Linear Regression Coefficient Paths')

# Logistic Regression
X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
fit_logistic = LogNet().fit(X, y)

# Cross-validation
cvfit = LogNet().fit(X, y)
_, cvpath = cvfit.cross_validation_path(X, y, cv=5)
ax = cvpath.plot(score='Binomial Deviance')
```

## Available Models

- **GaussNet**: Linear regression (Gaussian family)
- **LogNet**: Logistic regression (Binomial family)
- **MultiClassNet**: Multinomial regression
- **FishNet**: Poisson regression
- **MultiGaussNet**: Multi-response linear regression
- **CoxNet**: Cox proportional hazards model

## Key Features

### Elastic Net Regularization
```python
# Lasso (α=1, default)
fit_lasso = GaussNet(alpha=1.0).fit(X, y)

# Ridge (α=0)
fit_ridge = GaussNet(alpha=0.0).fit(X, y)

# Elastic Net (α=0.5)
fit_elastic = GaussNet(alpha=0.5).fit(X, y)
```

### Cross-Validation
```python
# Perform cross-validation
cvfit = GaussNet().fit(X, y)
_, cvpath = cvfit.cross_validation_path(X, y, cv=5)

# Get optimal lambda
lambda_min = cvpath.index_best['Mean Squared Error']
coef_min, intercept_min = cvfit.interpolate_coefs(lambda_min)
```

### Prediction and Interpolation
```python
# Predict at specific lambda values
predictions = fit_gaussian.predict(X_new, interpolation_grid=[0.1, 0.05])

# Interpolate coefficients
coef, intercept = fit_gaussian.interpolate_coefs(0.5)
```

### Weights and Offsets
```python
import pandas as pd

# Create DataFrame with response, weights, and offsets
Df = pd.DataFrame({
    'response': y,
    'weight': sample_weights,
    'offset': offsets
})

# Fit with weights and offsets
fit = GaussNet(response_id="response", 
               weight_id="weight", 
               offset_id="offset").fit(X, Df)
```

## Dependencies

- **Core**: numpy, scipy, pandas, scikit-learn, joblib, statsmodels, coxdev, tqdm
- **Build**: meson-python, meson, ninja, pybind11, setuptools_scm
- **Optional**: matplotlib (plotting), pytest (tests)

## Development

### Building from Source
See [Installation](#installation); `uv sync` builds the C++ extensions in place and
rebuilds them automatically on import after source changes.

### Running Tests
```bash
# Run all tests (or prefix with `uv run`)
pytest tests/

# Quicker subset
pytest tests --test-size=small

# Run specific test suites
pytest tests/paths/
pytest tests/compare_R/
```

### Saved Results in the Docs
The calibration experiments in the docs' Inference chapter are too slow to run
when the docs are built, so they are run locally and their results saved in
`docs/results/*.csv`, each with a header recording the commit and the package
versions it was computed with. To recompute all of them, commit your changes,
then, from the top of the repository, with the development install (`uv sync`,
or `pip install -e .` and the `doc` extra),
```bash
uv run python docs/experiments.py --all    # or: python docs/experiments.py --all
git add docs/results && git commit -m "docs: recompute the saved results"
```
`python docs/experiments.py probit_regression` recomputes one page. See
`docs/experiments.py` for the conventions.

**When changing versions** (a glmstar release, a new `lassoinf` or other
dependency pin, or the `src/glmnetpp` submodule), recompute these results, so
that the docs show results from the code they describe.

## Citation

If you use this package in your research, please cite:

```
Friedman, J., Hastie, T., & Tibshirani, R. (2010). 
Regularization paths for generalized linear models via coordinate descent. 
Journal of Statistical Software, 33(1), 1-22.
```

## License

GPL-2, as the R package `glmnet` and its C++ core `glmnetpp` (`src/glmnetpp`): see [LICENSE.txt](LICENSE.txt).

## Authors

- **Trevor Hastie** (hastie@stanford.edu)  
- **Balasubramanian Narasimhan** (naras@stanford.edu)
- **Jonathan Taylor** (jonathan.taylor@stanford.edu)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This package is based on the original R glmnet package by Jerome Friedman, Trevor Hastie, Rob Tibshirani, and others. The C++ implementation uses the Eigen library for linear algebra operations.
