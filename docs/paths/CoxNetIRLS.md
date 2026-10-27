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

# CoxNetIRLS

The `CoxNetIRLS` class fits a Cox proportional hazards model with elastic net regularization. It is suitable for survival (time-to-event) analysis with censored data. The path is computed by IRLS in Python around the generic `GLMNet` solver; see [CoxNet](CoxNet.md) for the C++ Cox path used by R's `glmnet`.

**Data requirements:**
> The `CoxNetIRLS` class fits Cox proportional hazards models for survival analysis. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a pandas DataFrame whose columns are named by their role:
>   - `'stop'`: event or observed time (float, > 0)
>   - `'status'`: event indicator (1=event, 0=censored)
>   - optionally `'start'` for start-stop (interval) data, `'strata'` for stratified
>     models, and `'weight'` and `'offset'`.
>
>   modelmatrix's `Outcome` builds it from other names, e.g.
>   `Outcome(Surv('time', 'status'), strata='inst').fit_transform(D)`.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_survival
from glmnet.cox import CoxFamily, CoxNetIRLS

X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
model = CoxNetIRLS(family=CoxFamily(tie_breaking='efron'))   # y: start, stop, status
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

The full API reference for `CoxNetIRLS` (all parameters, attributes and methods) is at
[CoxNetIRLS API](https://jonathan-taylor.github.io/glmstar/api/CoxNetIRLS.html).
