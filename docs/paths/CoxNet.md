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
>   - `y`: a pandas DataFrame whose columns the estimator's `outcome` names, a
>     modelmatrix `Surv(time, status)` (or `Surv(start, stop, status)` for start-stop
>     data) or an `Outcome` of it with `strata=`, `weight=` and `offset=` columns:
>   - the time: event or observed time (float, > 0)
>   - the status: event indicator (1=event, 0=censored)

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_survival
from glmnet.cox import CoxFamily
from glmnet.paths.coxnet import CoxNet
from modelmatrix import Surv

X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
model = CoxNet(family=CoxFamily(tie_breaking='efron'),
               outcome=Surv('start', 'event', 'status'))
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

The full API reference for `CoxNet` (all parameters, attributes and methods) is at
[CoxNet API](https://jonathan-taylor.github.io/glmstar/api/CoxNet.html).
