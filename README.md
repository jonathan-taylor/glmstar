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

### Prerequisites

The package requires the Eigen C++ library. You have two options:

#### Option 1: Use the included submodule (Recommended)
```bash
# Clone with submodules
git clone --recursive https://github.com/jonathan-taylor/pyglmnet.git
cd pyglmnet

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

#### Option 2: Install Eigen separately
```bash
# Clone Eigen 3.4.0
git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git --depth 5

# Set environment variable
export EIGEN_LIBRARY_PATH=/path/to/eigen
```

### Install the package

```bash
# Install in development mode
pip install .[dev]
pip install -e . --no-isolation-build

# Or install directly
pip install .
```

### Run the tests

```bash
pip install .[test]
pytest tests --test-size=small
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

- **Core**: numpy, scipy, pandas, scikit-learn
- **Build**: pybind11, setuptools, wheel
- **Optional**: matplotlib, joblib, statsmodels, tqdm

## Development

### Building from Source
```bash
# Install build dependencies
pip install -r requirements.txt

# Build the package
python setup.py build_ext --inplace
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/paths/
pytest tests/compare_R/
```

## Citation

If you use this package in your research, please cite:

```
Friedman, J., Hastie, T., & Tibshirani, R. (2010). 
Regularization paths for generalized linear models via coordinate descent. 
Journal of Statistical Software, 33(1), 1-22.
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Authors

- **Trevor Hastie** (hastie@stanford.edu)  
- **Balasubramanian Narasimhan** (naras@stanford.edu)
- **Jonathan Taylor** (jonathan.taylor@stanford.edu)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This package is based on the original R glmnet package by Jerome Friedman, Trevor Hastie, Rob Tibshirani, and others. The C++ implementation uses the Eigen library for linear algebra operations.
