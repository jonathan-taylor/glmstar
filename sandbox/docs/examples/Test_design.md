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

Implementing this code here: https://github.com/trevorhastie/glmnet/blob/3b268cebc7a04ff0c7b22931cb42b4c328ede307/R/glmnetFlex.R#L217C1-L233C6

'''
# standardize x if necessary
    if (intercept) {
        xm <- meansd$mean
    } else {
        xm <- rep(0.0, times = nvars)
    }
    if (standardize) {
        xs <- meansd$sd
    } else {
        xs <- rep(1.0, times = nvars)
    }
    if (!inherits(x, "sparseMatrix")) {
        x <- scale(x, xm, xs)
    }
'''

```{code-cell} ipython3
import numpy as np
from glmnet.base import Design
from glmnet import GLMNet
import rpy2
%load_ext rpy2.ipython

n, p = 100, 50
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
Y = rng.standard_normal(n)
beta = rng.standard_normal(p)
W = rng.uniform(1, 2, size=n)
W /= W.mean()
%R -i X,Y
```

# Without weights (all 1's)

## `standardize=True`, `intercept=True`

```{code-cell} ipython3
standardize, intercept = True, True
W = np.ones(n)
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * (Y - Y.mean()))).max() / n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize)
print(max(G$lambda)*nrow(X))
```

## `standardize=True`, `intercept=False`

```{code-cell} ipython3
standardize, intercept = True, False
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * Y)).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize)
print(max(G$lambda)*nrow(X))
```

## `standardize=False`, `intercept=True`

```{code-cell} ipython3
standardize, intercept = False, True
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * (Y - Y.mean()))).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize)
print(max(G$lambda)*nrow(X))
```

## `standardize=False`, `intercept=False`

```{code-cell} ipython3
standardize, intercept = False, False
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * Y)).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize)
print(max(G$lambda)*nrow(X))
```

# With weights

## `standardize=True`, `intercept=True`

```{code-cell} ipython3
standardize, intercept = True, True
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, W, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y, sample_weight=W)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * (Y - Y.mean()))).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize, weights=W)
print(max(G$lambda)*nrow(X))
```

## `standardize=True`, `intercept=False`

```{code-cell} ipython3
standardize, intercept = True, False
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, W, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y, sample_weight=W)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * Y)).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize, weights=W)
print(max(G$lambda)*nrow(X))
```

## `standardize=False`, `intercept=True`

```{code-cell} ipython3
standardize, intercept = False, True
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, W, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y, sample_weight=W)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * (Y - Y.mean()))).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize, weights=W)
print(max(G$lambda)*nrow(X))
```

## `standardize=False`, `intercept=False`

```{code-cell} ipython3
standardize, intercept = False, False
xm = (X * W[:,None]).sum(0) / W.sum()
x2 = (X**2 * W[:,None]).sum(0) / W.sum()
xs = np.sqrt(x2 - xm**2)
if not standardize:
    xs = np.ones_like(xs)
if not intercept:
    xm = np.zeros_like(xm)
X_ = (X - xm[None,:]) / xs[None,:]
D = Design(X, W, intercept=intercept, standardize=standardize)
assert np.allclose(D.centers_, xm)
assert np.allclose(D.scaling_, xs)
assert np.allclose(D @ np.hstack([0, beta]), X_ @ beta)
G = GLMNet(lambda_values=np.linspace(0.05,1,51)[::-1],
           fit_intercept=intercept, standardize=standardize)
G.fit(X, Y, sample_weight=W)
assert np.allclose(G.lambda_max_, np.fabs(X_.T @ (W * Y)).max()/n)
G.lambda_max_
```

```{code-cell} ipython3
%%R -i standardize,intercept,W
library(glmnet)
G=glmnet(X, Y, intercept=intercept, standardize=standardize, weights=W)
print(max(G$lambda)*nrow(X))
```
