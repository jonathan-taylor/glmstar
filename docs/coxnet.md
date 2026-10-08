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

# Regularized Cox Regression

This document parallels the R vignette [Regularized Cox Regression](https://glmnet.stanford.edu/articles/Coxnet.html), and describes how to fit regularized Cox models with `CoxNet`.

## Introduction

The Cox proportional hazards model is commonly used for the study of the relationship between predictor variables and survival time. In the usual survival analysis framework, we have data of the form $(y_1, x_1, \delta_1), \ldots, (y_n, x_n, \delta_n)$ where $y_i$, the observed time, is a time of failure if $\delta_i$ is 1 or a right-censored time if $\delta_i$ is 0. We also let $t_1 < t_2 < \ldots < t_m$ be the increasing list of unique failure times, and let $j(i)$ denote the index of the observation failing at time $t_i$.

The Cox model assumes a semi-parametric form for the hazard

$$
h_i(t) = h_0(t) e^{x_i^T \beta},
$$

where $h_i(t)$ is the hazard for patient $i$ at time $t$, $h_0(t)$ is a shared baseline hazard, and $\beta$ is a fixed, length $p$ vector. In the classic setting $n \geq p$, inference is made via the partial likelihood

$$
L(\beta) = \prod_{i=1}^m \frac{e^{x_{j(i)}^T \beta}}{\sum_{j \in R_i} e^{x_j^T \beta}},
$$

where $R_i$ is the set of indices $j$ with $y_j \geq t_i$ (those at risk at time $t_i$).

Note there is no intercept in the Cox model as it is built into the baseline hazard, and like it, would cancel in the partial likelihood.

`CoxNet` penalizes the negative log of the partial likelihood with an elastic net penalty. It uses the same C++ path algorithm as R's `glmnet(family = "cox")`, which computes the partial likelihood and its derivatives with [coxdev](https://github.com/jonathan-taylor/coxdev).

(Credits: The original `"coxnet"` algorithm for right-censored data was developed by Noah Simon, Jerome Friedman, Trevor Hastie and Rob Tibshirani. The other features for Cox models, introduced in R glmnet v4.1, were developed by Kenneth Tay, Trevor Hastie, Balasubramanian Narasimhan and Rob Tibshirani.)

## Basic usage for right-censored data

We use synthetic data for illustration. `X` must be an $n\times p$ matrix of covariate values --- each row corresponds to a patient and each column a covariate. The response `y` is a DataFrame with a column of failure/censoring times and a 0/1 status column, with 1 meaning the time is a failure time, and 0 a censoring time. `CoxFamily` names these columns (by default `'event'` and `'status'`) and chooses how ties are handled.

```{code-cell} ipython3
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg

from glmnet import CoxNet
from glmnet.cox import CoxFamily
from glmnet.data import make_survival

X, y, coef = make_survival(n_samples=100, n_features=20,
                           n_informative=5, snr=3.0,
                           random_state=42)
y.head()
```

We apply `CoxNet` to compute the solution path under default settings:

```{code-cell} ipython3
fit = CoxNet(family=CoxFamily(event_id='event', status_id='status')).fit(X, y)
```

All the standard options such as `alpha`, `weight_id`, `nlambda` and `standardize` apply, and their usage is similar to the Gaussian case (see [Quick Start](quick_start.md)).

We can plot the coefficients:

```{code-cell} ipython3
ax = fit.coef_path_.plot()
ax.set_title('Coefficient paths for Cox regression');
```

As before, we can extract the coefficients at certain values of $\lambda$:

```{code-cell} ipython3
coefs, _ = fit.interpolate_coefs(0.05)
pd.Series(coefs, index=fit.feature_names_in_).round(4)
```

Since the Cox model is not commonly used for prediction, we do not give an illustrative example of prediction. `fit.predict(X)` returns the linear predictor $x_i^T \hat\beta$ (the log relative risk) along the path; see `help(CoxNet.predict)`.

### Cross-validation

```{note}
Cross-validation for `CoxNet` (`cross_validation_path`) is not available yet: it is being fixed, and this section will then show the code. The description below is of R's `cv.glmnet`, which `cross_validation_path` will follow.
```

$K$-fold cross-validation (CV) for the Cox model is similar to that for other families except for two main differences.

First, the measures are the deviance (partial likelihood) and the Harrell *C index*. The C index is like the area under the curve (AUC) measure of concordance for survival data, but only considers comparable pairs. Pure concordance would record the fraction of pairs for which the order of the death times agree with the order of the predicted risk. However, with survival data, if an observation is right censored at a time *before* another observation's death time, they are not comparable. Unlike most error measures, a higher C index means better prediction performance.

Second, the grouped CV partial likelihood for the $K$th fold is obtained by subtraction, i.e. by subtracting the log partial likelihood evaluated on the full dataset from that evaluated on the $(K-1)/K$ dataset. This makes more efficient use of risk sets. Computing the log partial likelihood only on the $K$th fold is only reasonable if each fold has a large number of observations.

### Handling of ties

`CoxNet` supports both the Breslow and Efron approximations for handling tied survival times, chosen by `tie_breaking` in `CoxFamily`. The default is `'efron'`, matching statsmodels' `PHReg` and R's `survival::coxph`. (In R, `glmnet` chooses with `cox.ties`.)

With `lambda_values=[0]`, `CoxNet` fits the unpenalized Cox model, which we can compare with `PHReg` for data with many ties:

```{code-cell} ipython3
rng = np.random.default_rng(1)
nobs, nvars = 100, 15
x = rng.standard_normal((nobs, nvars))

# response with many ties
ty = np.repeat(rng.exponential(size=nobs // 5), 5)
tcens = rng.binomial(1, 0.3, size=nobs)
y = pd.DataFrame({'time': ty, 'status': tcens})

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, ties in zip(axes, ['efron', 'breslow']):
    family = CoxFamily(event_id='time', status_id='status', tie_breaking=ties)
    glmnet_fit = CoxNet(family=family, lambda_values=[0.]).fit(x, y)
    coxph_fit = PHReg(ty, x, status=tcens, ties=ties).fit()
    ax.scatter(glmnet_fit.coefs_[0], coxph_fit.params)
    ax.axline((0, 0), slope=1, color='gray', ls='--')
    ax.set_xlabel('CoxNet'); ax.set_ylabel('PHReg'); ax.set_title(f'{ties} ties')
```

## Cox models for start-stop data

`CoxNet` can fit models where the response is a (start, stop] time interval. As explained in Therneau & Grambsch (2000), the ability to work with start-stop responses opens the door to fitting regularized Cox models with

* time-dependent covariates,
* time-dependent strata,
* left truncation,
* multiple time scales,
* multiple events per subject,
* independent increment, marginal, and conditional models for correlated data, and
* various forms of case-cohort models.

The code below shows how to create a response of this type and fit such a model. The start times are given by `start_id`.

```{code-cell} ipython3
rng = np.random.default_rng(2)
xvec = rng.standard_normal(nobs * nvars)
xvec[rng.choice(nobs * nvars, size=int(0.4 * nobs * nvars), replace=False)] = 0
x = xvec.reshape((nobs, nvars))          # dense x
x_sparse = scipy.sparse.csc_matrix(x)     # sparse x

# start-stop response
beta = rng.standard_normal(5)
fx = x[:, :5] @ beta / 3
ty = rng.exponential(np.exp(-fx))
tcens = rng.binomial(1, 0.3, size=nobs)
starty = rng.uniform(size=nobs)
yss = pd.DataFrame({'start': starty, 'stop': starty + ty, 'status': tcens})

family = CoxFamily(event_id='stop', status_id='status', start_id='start')
fit = CoxNet(family=family).fit(x, yss)
```

The call above would have worked as well with `x_sparse` in place of `x`:

```{code-cell} ipython3
fit_sparse = CoxNet(family=family).fit(x_sparse, yss)
np.max(np.abs(fit.coefs_ - fit_sparse.coefs_))
```

As a sanity check, fitting start-stop responses with `lambda_values=[0]` matches `PHReg`, which takes the start times as `entry`:

```{code-cell} ipython3
glmnet_fit = CoxNet(family=family, lambda_values=[0.]).fit(x, yss)
coxph_fit = PHReg(yss['stop'], x, status=tcens, entry=starty).fit()
fig, ax = plt.subplots()
ax.scatter(glmnet_fit.coefs_[0], coxph_fit.params)
ax.axline((0, 0), slope=1, color='gray', ls='--')
ax.set_xlabel('CoxNet'); ax.set_ylabel('PHReg');
```

## Stratified Cox models

One extension of the Cox regression model is to allow for strata that divide the observations into disjoint groups. Each group has its own baseline hazard function, but the groups share the same coefficient vector for the covariates provided by the design matrix `x`.

`CoxNet` can fit stratified Cox models with the elastic net penalty. In R, `glmnet` attaches strata to the response with `stratifySurv`; here we add a strata column to the response DataFrame and name it with `strata_id`. The labels can be of any type.

```{code-cell} ipython3
strata = np.tile(np.arange(1, 6), nobs // 5)
y2 = y.assign(strata=strata)
y2.head(6)
```

```{code-cell} ipython3
family = CoxFamily(event_id='time', status_id='status', strata_id='strata')
fit = CoxNet(family=family).fit(x, y2)
```

With `lambda_values=[0]` the stratified fit matches `PHReg` with `strata`:

```{code-cell} ipython3
x = rng.standard_normal((nobs, nvars))
glmnet_fit = CoxNet(family=family, lambda_values=[0.]).fit(x, y2)
coxph_fit = PHReg(y2['time'], x, status=y2['status'], strata=strata).fit()
np.max(np.abs(glmnet_fit.coefs_[0] - coxph_fit.params))
```

## Survival curves

R's `glmnet` provides a `survfit` method for Cox fits, which estimates the baseline hazard and plots survival curves. This is not yet available in Python; `predict` returns the linear predictor, from which relative risks $e^{x^T \hat\beta}$ can be computed:

```{code-cell} ipython3
fit = CoxNet(family=CoxFamily(event_id='time', status_id='status')).fit(x, y)
fit.predict(x[:3], interpolation_grid=0.05)
```

To be consistent with other methods, if `interpolation_grid` is not specified, predictions are returned for the entire $\lambda$ sequence:

```{code-cell} ipython3
fit.predict(x[:3]).shape
```

## `CoxNetIRLS`

`CoxNetIRLS` fits the same model by IRLS in Python around the generic `GLMNet` solver (see [GLM families](glmnet_family.md)). `CoxNet` is faster and matches R's `glmnet`; see [CoxNetIRLS](paths/CoxNetIRLS.md).

## References

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

2. Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5), 1-13.

3. Therneau, T. M., & Grambsch, P. M. (2000). *Modeling survival data: extending the Cox model*. Springer Science & Business Media.

---

*This document adapts the R glmnet vignette for the Python glmnet package. The original R vignette was written by Kenneth Tay, Noah Simon, Jerome Friedman, Trevor Hastie, Rob Tibshirani, and Balasubramanian Narasimhan.*
