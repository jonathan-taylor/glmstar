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

# Penalty factors (and excluding variables)

A **penalty factor** multiplies the penalty on one variable:

- factor 1: the usual penalty;
- factor 0: the variable is not penalized;
- **infinite factor: the variable is excluded.** In `glmnet`, an infinite
  penalty factor is how a variable is excluded.

R's `glmnet` has separate `exclude` and `penalty.factor` arguments, and
each can be fixed or a function of the data. `exclude` has accepted a
function since glmnet 4.1-2, and `penalty.factor` since 5.1. In
`cv.glmnet` the function is called again on the training data of each
fold, so the held-out rows never influence which variables are dropped or
how they are penalized.

In `glmnet`:

| R | `glmnet` |
|---|---|
| `penalty.factor = pf` (fixed) | `penalty_factor=pf` |
| `exclude = idx` (fixed) | `exclude=idx`, the same as giving those variables `np.inf` in `penalty_factor` |
| `penalty.factor = function(x, y, ...)` | override `get_penalty_factor(X, y)` |
| `exclude = function(x, y, ...)` | override `get_penalty_factor(X, y)`, returning `np.inf` for the variables to exclude |

So **R's `exclude` and `penalty.factor` are both handled by one method,
`get_penalty_factor`.** It returns penalty factors as for
`penalty_factor=`, and exclusions are the infinite ones. Like R's
functions, it is called at the start of each `fit`, so it reruns on each
training fold in cross-validation.

```{note}
`prefilter(X, y)`, which returned the indices of the variables to exclude,
is deprecated. A `prefilter` override still works, with a `FutureWarning`;
its indices get an infinite penalty factor.
```

```{code-cell} ipython3
from dataclasses import dataclass
import numpy as np
from glmnet import GaussNet
```

## Data with mostly-zero columns

We simulate 100 observations on 20 features. About half the entries of
each column are zero, and three noise columns (3, 12 and 17) are almost
entirely zero.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 100, 20
X = rng.standard_normal((n, p))
X[rng.uniform(size=(n, p)) < 0.5] = 0
sparse_cols = [3, 12, 17]
X[:, sparse_cols] *= rng.uniform(size=(n, 3)) < 0.1
beta = np.zeros(p)
beta[[0, 1, 5, 7]] = [2, -1.5, 3, 2]
y = X @ beta + rng.standard_normal(n)
np.round((X == 0).mean(0), 2)
```

## A fixed exclude list

The equivalent of `glmnet(x, y, exclude = c(4, 13, 18))` in R. R
indexes from 1, so the Python indices are one smaller.

```{code-cell} ipython3
fit_static = GaussNet(exclude=[3, 12, 17]).fit(X, y)
```

## `exclude` as a function

The R glmnet vignette gives this as a typical example: it drops every
column that is zero in more than 80% of the observations.

```r
filter <- function(x, ...) which(colMeans(x == 0) > 0.8)
fit <- glmnet(x, y, exclude = filter)
cvfit <- cv.glmnet(x, y, exclude = filter)
```

In Python, `get_penalty_factor` gives those columns an infinite factor.
The cutoff is a dataclass field, so it can be set in the constructor and
is kept by `sklearn.base.clone`, which cross-validation uses to refit the
model on each fold.

```{code-cell} ipython3
@dataclass
class SparseFilterGaussNet(GaussNet):

    max_zero_frac: float = 0.8

    def get_penalty_factor(self, X, y):
        X = np.asarray(X)
        return np.where((X == 0).mean(0) > self.max_zero_frac, np.inf, 1.)

fit_filter = SparseFilterGaussNet().fit(X, y)
fit_filter.excluded_
```

The variables with an infinite factor are listed in `excluded_`, together
with any given in `exclude=`. On the full data this filter picks the same
three columns as the fixed list, so the path is the same:

```{code-cell} ipython3
np.abs(fit_filter.coefs_ - fit_static.coefs_).max()
```

The excluded coefficients stay at zero along the whole path:

```{code-cell} ipython3
ax = fit_filter.coef_path_.plot()
```

### Cross-validation

This is the analogue of `cv.glmnet(x, y, exclude = filter)`.
`cross_validation_path` clones the estimator and fits it on each training
fold, so `get_penalty_factor` runs again on each fold's training rows. To
show this, the subclass below prints which columns it drops on each
call:

```{code-cell} ipython3
@dataclass
class LoggedFilterGaussNet(SparseFilterGaussNet):

    def get_penalty_factor(self, X, y):
        pf = super().get_penalty_factor(X, y)
        print(f'{X.shape[0]} rows: excluding {np.nonzero(np.isinf(pf))[0].tolist()}')
        return pf

cvfit = LoggedFilterGaussNet().fit(X, y)
_, cvpath = cvfit.cross_validation_path(X, y, cv=5)
```

The first line is the fit on all 100 rows. Each of the other five is the
fit on an 80-row training fold.

```{code-cell} ipython3
ax = cvpath.plot(score='Mean Squared Error')
```

If you instead want the variables chosen once, on all the data, compute
the factors yourself and pass them as `penalty_factor=` (or the indices
as `exclude=`). Constructor arguments are copied to each fold as they are.

## `penalty.factor` as a function

The adaptive lasso divides each variable's penalty by the size of its
least squares coefficient. In R 5.1:

```r
pf <- function(x, y, ...) 1 / abs(coef(lm(y ~ x))[-1])
fit <- glmnet(x, y, penalty.factor = pf)
cvfit <- cv.glmnet(x, y, penalty.factor = pf)
```

This is the same method:

```{code-cell} ipython3
@dataclass
class AdaptiveGaussNet(GaussNet):

    def get_penalty_factor(self, X, y):
        X1 = np.column_stack([np.ones(X.shape[0]), X])
        beta_ls = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
        return 1 / np.fabs(beta_ls)

fit_adaptive = AdaptiveGaussNet().fit(X, y)
np.round(fit_adaptive.penalty_factor_, 2)
```

The constructor's `penalty_factor` is left unchanged. The factors used
for the last fit are in `penalty_factor_`.

```{code-cell} ipython3
_, cvpath_adaptive = fit_adaptive.cross_validation_path(X, y, cv=5)
ax = cvpath_adaptive.plot(score='Mean Squared Error')
```

## Both at once

Because exclusions are infinite factors, one method can do both: here,
adaptive penalty factors for the variables that pass the sparsity
filter.

```{code-cell} ipython3
@dataclass
class FilteredAdaptiveGaussNet(GaussNet):

    max_zero_frac: float = 0.8

    def get_penalty_factor(self, X, y):
        X = np.asarray(X)
        X1 = np.column_stack([np.ones(X.shape[0]), X])
        beta_ls = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
        pf = 1 / np.fabs(beta_ls)
        pf[(X == 0).mean(0) > self.max_zero_frac] = np.inf
        return pf

fit_both = FilteredAdaptiveGaussNet().fit(X, y)
fit_both.excluded_
```

## Notes

- The same approach works for any `*Net` estimator (`LogNet`, `FishNet`,
  `CoxNet`, `MultiGaussNet`, ...) and for `GLMNet`. All of them call
  `self.get_penalty_factor(X, y)` at the start of `fit`.
- The default `get_penalty_factor` returns the constructor's
  `penalty_factor`. An override replaces it, unless it starts from
  `super().get_penalty_factor(X, y)`. Indices given in `exclude=` are
  always excluded as well.
- `get_penalty_factor` receives `X` and `y` exactly as they were passed
  to `fit`, so `y` may be a DataFrame that also holds the weight or offset
  columns. R's functions receive `x`, `y` and `weights`. To use the
  weights in Python, read them from `y` through `weight_id`.
- `penalty_factor_` holds the factors the solvers used, with each
  infinite factor replaced by 1 (as R does internally). The variables
  with infinite factors are listed in `excluded_`.
