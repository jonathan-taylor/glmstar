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

# Selective inference after a logistic lasso

Once the lasso has chosen a set of variables, the usual confidence intervals
and p-values for their coefficients are no longer valid: the same data chose
the variables and estimate their effects. Selective inference accounts for the
selection.

`glmnet.inference` extracts what is needed from a fit: the problem the lasso
solved and the distribution of the score used to select. The inference itself
is done by the separate
[`lassoinf`](https://github.com/jonathan-taylor/lassoinf) package.

```{code-cell} ipython3
%pip install lassoinf
```

## A logistic lasso

We simulate $n=500$ observations of $p=10$ features, of which only the first
three affect the response, and fit the lasso path with `LogNet`. The inference
assumes the fit solves the lasso problem exactly, so we tighten the
convergence threshold.

```{code-cell} ipython3
import numpy as np
from scipy.special import expit

from glmnet import LogNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.inference import glmstar_inference_problem
from lassoinf import LassoInference

rng = np.random.default_rng(0)
n, p = 500, 10
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = [1.0, -0.8, 0.6]
y = rng.binomial(1, expit(X @ beta))

fit = LogNet(control=FastNetControl(thresh=1e-14)).fit(X, y)
```

We pick one value of $\lambda$ on the path. These variables are selected:

```{code-cell} ipython3
lam = fit.lambda_values_[15]
np.nonzero(fit.coefs_[15])[0]
```

## Inference after selection

`glmstar_inference_problem` describes the selection at `lam`. All the data
were used to select, so there is no randomization (`scalar_noise` is 0).
`LassoInference` gives confidence intervals and p-values for the selected
coefficients that account for their selection. The first row is the
intercept.

```{code-cell} ipython3
info = glmstar_inference_problem(fit, X, y, lambda_val=lam)
inference = LassoInference(**info.lasso_args(), level=0.95)
inference.summary_
```

See the [`lassoinf` documentation](https://github.com/jonathan-taylor/lassoinf)
for other families, weights and offsets, and for carving (holding out data
from selection).
