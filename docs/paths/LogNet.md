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

# LogNet

The `LogNet` class fits a logistic regression model with elastic net regularization (binomial family). It is suitable for binary classification problems.

**Data requirements:**
> The `LogNet` class fits logistic regression models (binomial family) with elastic net regularization. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a 1D array, Series, or DataFrame column of shape `(n_samples,)` containing binary labels. Labels can be 0/1, True/False, or two unique string values (e.g., 'A'/'B'). If string labels are provided, they will be automatically encoded.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_dataset
from glmnet.paths.lognet import LogNet

X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
model = LogNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.lognet.LogNet
    :members:
    :inherited-members:
    :show-inheritance:
```
