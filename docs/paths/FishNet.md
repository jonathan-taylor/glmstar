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

# FishNet

The `FishNet` class fits a Poisson regression model with elastic net regularization (Poisson family). It is suitable for count data and event rate modeling.

**Data requirements:**
> The `FishNet` class fits Poisson regression models for count data. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a 1D array, Series, or DataFrame column of shape `(n_samples,)` containing non-negative integer counts (e.g., number of events).

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.fishnet import FishNet

X, y, coef, intercept = make_dataset(FishNet, n_samples=100, n_features=10, snr=5)
model = FishNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.fishnet.FishNet
    :members:
    :inherited-members:
    :show-inheritance:
```
