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

# Excluding variables

In R, `glmnet` has an `exclude` argument. It can be a fixed set of
column indices, or a function of the data that returns the indices to
drop. The R glmnet vignette gives this as a typical example: it drops
every column that is zero in more than 80% of the observations.

```r
filter <- function(x, ...) which(colMeans(x == 0) > 0.8)
fit <- glmnet(x, y, exclude = filter)
cvfit <- cv.glmnet(x, y, exclude = filter)
```

When `exclude` is a function, `cv.glmnet` calls it again on the
training data of each fold. The filtering step is then part of what
cross-validation evaluates, and the held-out data never decides which
columns are dropped.

In `glmnet`, a fixed set of columns is passed with `exclude=`. For the
data-dependent version, subclass a `*Net` estimator and override its
`prefilter(X, y)` method. The indices `prefilter` returns (0-based) are
added to `exclude` each time `fit` is called.

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

## A filter function, via subclassing

R's `filter` becomes the body of `prefilter`. The cutoff is a dataclass
field, so it can be set in the constructor and is kept by
`sklearn.base.clone`, which cross-validation uses to refit the model on
each fold.

```{code-cell} ipython3
@dataclass
class SparseFilterGaussNet(GaussNet):

    max_zero_frac: float = 0.8

    def prefilter(self, X, y):
        X = np.asarray(X)
        return np.nonzero((X == 0).mean(0) > self.max_zero_frac)[0]

fit_filter = SparseFilterGaussNet().fit(X, y)
fit_filter.excluded_
```

On the full data this filter picks the same three columns, so the path
is the same as with the fixed list:

```{code-cell} ipython3
np.abs(fit_filter.coefs_ - fit_static.coefs_).max()
```

The excluded coefficients stay at zero along the whole path:

```{code-cell} ipython3
ax = fit_filter.coef_path_.plot()
```

## Cross-validation

This is the analogue of `cv.glmnet(x, y, exclude = filter)`.
`cross_validation_path` clones the estimator and fits it on each
training fold, so `prefilter` runs again on each fold's training rows.
To show this, the subclass below records which columns it drops on
each call:

```{code-cell} ipython3
@dataclass
class LoggedFilterGaussNet(SparseFilterGaussNet):

    def prefilter(self, X, y):
        excluded = super().prefilter(X, y)
        print(f'{X.shape[0]} rows: excluding {excluded.tolist()}')
        return excluded

cvfit = LoggedFilterGaussNet().fit(X, y)
_, cvpath = cvfit.cross_validation_path(X, y, cv=5)
```

The first line is the fit on all 100 rows. Each of the other five is
the fit on an 80-row training fold.

```{code-cell} ipython3
ax = cvpath.plot(score='Mean Squared Error')
```

## Penalty factors as a function

Since version 5.1, R's glmnet also accepts a function for
`penalty.factor`. For example, the adaptive lasso divides each
variable's penalty by the size of its least squares coefficient:

```r
pf <- function(x, y, ...) 1 / abs(coef(lm(y ~ x))[-1])
fit <- glmnet(x, y, penalty.factor = pf)
cvfit <- cv.glmnet(x, y, penalty.factor = pf)
```

The Python equivalent is to override `get_penalty_factor(X, y)`. As
with `prefilter`, it is called at the start of each `fit`, so
cross-validation recomputes the factors on each training fold. The
constructor's `penalty_factor` is left unchanged; the factors used for
the last fit are stored in `penalty_factor_`.

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

```{code-cell} ipython3
_, cvpath_adaptive = fit_adaptive.cross_validation_path(X, y, cv=5)
ax = cvpath_adaptive.plot(score='Mean Squared Error')
```

## Notes

- The same approach works for any `*Net` estimator (`LogNet`,
  `FishNet`, `CoxNet`, `MultiGaussNet`, ...) and for `GLMNet`. All of
  them call `self.prefilter(X, y)` and `self.get_penalty_factor(X, y)`
  at the start of `fit`.
- `get_penalty_factor` returns factors as for `penalty_factor=`; an
  infinite factor excludes the variable.
- `prefilter` receives `X` and `y` exactly as they were passed to
  `fit`, so `y` may be a DataFrame that also holds the weight or offset
  columns. R's filter function receives `x`, `y` and `weights`; to use
  the weights in Python, read them from `y` through `weight_id`.
- `prefilter` returns 0-based column indices. They are combined with
  any indices given in `exclude=`.
- Excluding a column has the same effect as giving it an infinite
  penalty factor.
