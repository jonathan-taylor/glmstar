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

# Selective inference with `lassoinf`

Once the lasso has chosen a set of variables, the usual confidence intervals
and p-values for their coefficients are no longer valid: the same data chose
the variables and estimate their effects. Selective inference accounts for the
selection. These methods live in the separate
[`lassoinf`](https://github.com/jonathan-taylor/lassoinf) package, which takes
a fitted `glmstar` model directly.

```{code-cell} ipython3
%pip install lassoinf
```

## A logistic lasso

We simulate $n=500$ observations of $p=10$ features, of which only the first
three affect the response.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl
from lassoinf import glmstar_inference

rng = np.random.default_rng(0)
n, p = 500, 10
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = [1.0, -0.8, 0.6]
df = pd.DataFrame({'y': rng.binomial(1, expit(X @ beta))})
```

We fit the lasso path. `lassoinf` checks that the solution satisfies the KKT
conditions of the lasso problem, so we tighten the convergence threshold.

```{code-cell} ipython3
fit = GLMNet(family=sm.families.Binomial(),
             response_id='y',
             control=GLMNetControl(thresh=1e-14))
fit.fit(X, df)
lam = fit.lambda_values_[15]
np.nonzero(fit.coefs_[15])[0]
```

## Inference after selection

`glmstar_inference` gives confidence intervals and p-values for the
coefficients of the variables selected at `lam` (the first row is the
intercept), accounting for their selection.

```{code-cell} ipython3
inference = glmstar_inference(fit, X, df, lambda_val=lam)
inference.summary_
```

## Carving

Holding out some of the data from selection gives more powerful inference.
Here the lasso selects on a random 70% of the observations, and inference uses
all of them; `selection_rows` tells `glmstar_inference` which rows were used
to select.

```{code-cell} ipython3
rows = rng.choice(n, int(0.7 * n), replace=False)
fit_sel = GLMNet(family=sm.families.Binomial(),
                 response_id='y',
                 control=GLMNetControl(thresh=1e-14))
fit_sel.fit(X[rows], df.iloc[rows])
lam = fit_sel.lambda_values_[15]

carved = glmstar_inference(fit_sel, X, df, lambda_val=lam, selection_rows=rows)
carved.summary_
```

See the [`lassoinf` documentation](https://github.com/jonathan-taylor/lassoinf)
for other families, weights and offsets, and other targets of inference.
