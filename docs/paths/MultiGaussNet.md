---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# MultiGaussNet

The `MultiGaussNet` class fits a multi-response linear regression model with elastic net regularization (Gaussian family). It is suitable for continuous response data with multiple outputs.

**Data requirements:**
> The `MultiGaussNet` class fits multi-response (multi-output) linear regression models. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a 2D array or DataFrame of shape `(n_samples, n_targets)` (multiple continuous responses).

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.multigaussnet import MultiGaussNet

X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=100, n_features=10, 
                                    n_targets=3, snr=5)
model = MultiGaussNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.multigaussnet.MultiGaussNet
    :members:
    :inherited-members:
    :show-inheritance:
```
