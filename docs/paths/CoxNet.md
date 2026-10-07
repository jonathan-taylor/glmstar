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

# CoxNet

The `CoxNet` class fits a Cox proportional hazards model with elastic net regularization using the C++ Cox path from `glmnetpp` (via `coxdev`), the same code used by R's `glmnet(family="cox")`. It supports Breslow or Efron tie-breaking, right-censored or start-stop data, strata, sample weights and offsets, and dense or sparse `X`.

**Data requirements:**
> The `CoxNet` class fits Cox proportional hazards models for survival analysis. It expects:
> - `X`: a 2D NumPy array, pandas DataFrame or scipy sparse matrix of shape `(n_samples, n_features)` (predictors).
> - `y`: a pandas DataFrame whose columns are named by `family` (a `CoxFamily`):
>   - `event_id` (default `'event'`): event or observed time (float, > 0)
>   - `status_id` (default `'status'`): event indicator (1=event, 0=censored)
>   - optionally `start_id` for start-stop (interval) data, and `strata_id` for stratified models
>
>   along with any columns named by `weight_id` and `offset_id`.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_survival
from glmnet.cox import CoxFamily
from glmnet.paths.coxnet import CoxNet

X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
model = CoxNet(family=CoxFamily(start_id='start', tie_breaking='efron'))
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.paths.coxnet.CoxNet
    :members:
    :inherited-members:
    :show-inheritance:
```
