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

# GaussNet

The `GaussNet` class fits a linear regression model with elastic net regularization (Gaussian family). It is suitable for continuous response data.

**Data requirements:**
> The `GaussNet` class fits linear regression models with elastic net regularization. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (continuous predictors).
> - `y`: a 1D NumPy array, pandas Series, or DataFrame column of shape `(n_samples,)` (continuous response).
> The response should be numeric (float or int).

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.gaussnet import GaussNet

X, y, coef, intercept = make_dataset(GaussNet, n_samples=100, n_features=10, snr=5)
model = GaussNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.gaussnet.GaussNet
    :members:
    :inherited-members:
    :show-inheritance:
```
