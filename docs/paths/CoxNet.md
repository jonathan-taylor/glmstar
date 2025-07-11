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

The `CoxNet` class fits a Cox proportional hazards model with elastic net regularization. It is suitable for survival (time-to-event) analysis with censored data.

**Data requirements:**
> The `CoxNet` class fits Cox proportional hazards models for survival analysis. It expects:
> - `X`: a 2D NumPy array or pandas DataFrame of shape `(n_samples, n_features)` (predictors).
> - `y`: a pandas DataFrame with at least two columns:
>   - `'event'`: event or observed time (float)
>   - `'status'`: event indicator (1=event, 0=censored)
>   Optionally, a `'start'` column for start-stop (interval) data, and a `'strata'` column for stratified models.

## Example Usage

```{code-cell} ipython3
from glmnet.data import make_survival
from glmnet.cox import CoxNet

X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
model = CoxNet()
model.fit(X, y)
print(model.coefs_.shape)
```

## API Reference

```{eval-rst}
.. autoclass:: glmnet.cox.CoxNet
    :members:
    :inherited-members:
    :show-inheritance:
```
