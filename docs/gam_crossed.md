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

# Crossed random effects at scale

Ratings of items by customers are the motivating example of Ghosh, Hastie
and Owen (2022): a regression

$$y_{ij} = x_{ij}^T\beta + a_i + b_j + e_{ij},\qquad a_i \sim N(0, \sigma_A^2),\
b_j \sim N(0, \sigma_B^2),\ e_{ij} \sim N(0, \sigma_E^2),$$

for customer $i$ and item $j$, observed for a sparse, very unbalanced set
of pairs. With $N$ observations, generalized least squares (GLS) for
$\beta$, or REML for the variances, costs at least $O(N^{3/2})$ in
general: the random effects are crossed, so no ordering of them makes the
mixed model equations' factorization sparse. Ghosh, Hastie and Owen's
algorithm costs $O(N)$:

1. estimate the variances by the method of moments (Gao and Owen, 2017),
   from the residuals of the fit without the random effects;
2. with them fixed, compute $\hat\beta_{\text{GLS}}$ by backfitting: each
   random effect's update is a vector of shrunken group means;
3. its standard errors, from backfitting the columns of $X$.

In `glmnet.gam` these are `method=CrossedMoments()`, `GAM`'s backfitting,
and `fixed_effects_`. None of the three forms the $N \times P$ design or a
$P \times P$ matrix.

```{code-cell} ipython3
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from glmnet.gam import GAM, GAMControl, re, linear, CrossedMoments, ExperimentalWarning

warnings.simplefilter('ignore', ExperimentalWarning)
```

## Simulated ratings

`ratings(N, R, C)` draws $N$ ratings by $R$ customers of $C$ items. Each
rating's customer is uniform, and its item follows a Zipf law, so that a
few items are rated very often and most rarely, as in real ratings data.
There are two covariates:

- `x`, a customer's attribute (most of its variance is between
  customers);
- `z`, an attribute of the rating.

The true values are $\beta = (0.3, -0.2)$, $\sigma_A = 1$, $\sigma_B =
0.4$ and $\sigma_E = 2$.

```{code-cell} ipython3
def ratings(N, R, C, seed=0):
    rng = np.random.default_rng(seed)
    customer = rng.integers(0, R, N)
    item = (rng.zipf(1.4, N) - 1) % C
    a, b = rng.normal(0, 1.0, R), rng.normal(0, 0.4, C)
    x = rng.standard_normal(R)[customer] + 0.5 * rng.standard_normal(N)
    z = rng.standard_normal(N)
    y = 5 + 0.3 * x - 0.2 * z + a[customer] + b[item] + rng.normal(0, 2, N)
    return pd.DataFrame({'customer': customer, 'item': item, 'x': x, 'z': z, 'y': y})

D = ratings(100000, 10000, 2000)
counts = D['item'].value_counts()
counts.describe()
```

The items' numbers of ratings range from one to many thousands.

## The fit

The model is two linear terms and two crossed random intercepts:

```{code-cell} ipython3
terms = [linear('x'), linear('z'), re('customer'), re('item')]
start = time.time()
fit = GAM(terms=terms, method=CrossedMoments()).fit(D, D['y'])
fit.variance_components_, time.time() - start
```

The variances' estimates are near the true values: 1 for the customers, 4
for the errors, and 0.16 for the items, which are the hardest to estimate
because most items have few ratings. The fixed effects, with their GLS
standard errors:

```{code-cell} ipython3
fit.fixed_effects_
```

### Ordinary least squares is naive

Ordinary least squares on `x` and `z` ignores the random effects. Its
estimates are unbiased here, but its standard error is too small for
`x`, whose variation is mostly between customers (Ghosh, Hastie and Owen
call this OLS's naivete). For `z`, which varies within customers, GLS is
the more precise (OLS's inefficiency):

```{code-cell} ipython3
ols = sm.OLS(D['y'], sm.add_constant(D[['x', 'z']])).fit()
pd.DataFrame({'OLS': ols.bse.values[1:], 'GLS': fit.fixed_effects_['std_error'].values[1:]},
             index=['x', 'z'])
```

The predicted effects (BLUPs) are each group's shrunken mean residual.
Shrinkage toward 0 is strongest for the items rated least:

```{code-cell} ipython3
items = fit.terms_[3].effects_
n = D['item'].value_counts().reindex(items.index)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(n, items.values, s=4, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('ratings of the item')
ax.set_ylabel('predicted item effect')
fig.tight_layout()
```

## The cost grows about linearly with $N$

The same model, with the numbers of customers and items growing with $N$,
timed for the fit and for the standard errors:

```{code-cell} ipython3
rows = []
for N in [25000, 50000, 100000, 200000, 400000]:
    D_N = ratings(N, N // 10, N // 50)
    start = time.time()
    G = GAM(terms=terms, method=CrossedMoments()).fit(D_N, D_N['y'])
    fit_time = time.time() - start
    start = time.time()
    se = G.fixed_effects_['std_error']
    rows.append({'N': N, 'customers': N // 10, 'items': N // 50,
                 'fit (s)': fit_time, 'standard errors (s)': time.time() - start,
                 'sweeps': G.regularizer_.n_sweeps_, 'se(x)': se.iloc[1]})
timing = pd.DataFrame(rows).set_index('N')
timing
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(timing.index, timing['fit (s)'], 'o-', label='fit')
ax.loglog(timing.index, timing['standard errors (s)'], 's-', label='standard errors')
ref = timing['fit (s)'].iloc[0] * timing.index / timing.index[0]
ax.loglog(timing.index, ref, 'k:', label='linear in N')
ax.set_xticks(timing.index, [f'{n:,}' for n in timing.index], rotation=30)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('N')
ax.set_ylabel('seconds')
ax.legend()
fig.tight_layout()
```

The standard errors' time is linear in $N$: a fixed number of backfitting
sweeps per column. The fit's grows a little faster, because its number of
sweeps (to a relative change of `bf_tol`, 1e-10) grows from about 100 to
about 200 here. Ghosh, Hastie and Owen bound the number of sweeps, for
backfitting crossed random effects, by a condition on the pattern of the
observations.

At 400,000 ratings the model has 48,000 random effects. A dense $P \times
P$ matrix of that size would take about 18 GB, and REML would factor it at
every evaluation of its criterion.

## Against REML

For a small problem the variances can also be estimated by REML, which
forms the dense $P \times P$ matrices:

```{code-cell} ipython3
small = ratings(4000, 400, 100)
start = time.time()
reml = GAM(terms=terms, method='REML').fit(small, small['y'])
reml_time = time.time() - start
start = time.time()
moments = GAM(terms=terms, method=CrossedMoments()).fit(small, small['y'])
moments_time = time.time() - start
pd.DataFrame({'REML': reml.variance_components_['variance'],
              'moments': moments.variance_components_['variance']}).assign(
    truth=[1, 0.16, 4]), reml_time, moments_time
```

The method of moments is less efficient than REML, but its estimates are
consistent (Gao and Owen, 2017), and at this size it is about a hundred
times faster. The fixed effects are close:

```{code-cell} ipython3
pd.concat({'REML': reml.fixed_effects_, 'moments': moments.fixed_effects_}, axis=1)
```

## What is not covered

- `CrossedMoments` is for a Gaussian response with equal prior weights, and
  random intercepts. Random slopes and non-Gaussian responses (Ghosh,
  Hastie and Owen's logistic regression with crossed random effects) are
  not yet available.
- `fixed_effects_` is for linear terms and blocks of columns; a factor
  among the fixed effects is given as a block of its indicator columns.
- Standard errors of predictions, `predict(..., return_std=True)`, form the
  dense covariance.

## References

- Gao, K. and Owen, A. B. (2017). Efficient moment calculations for
  variance components in large unbalanced crossed random effects models.
  *Electronic Journal of Statistics* 11(1), 1235–1296.
- Ghosh, S., Hastie, T. and Owen, A. B. (2022). Backfitting for large scale
  crossed random effects regressions. *The Annals of Statistics* 50(1).
  arXiv:2007.10612.
- Ghosh, S., Hastie, T. and Owen, A. B. (2022). Scalable logistic
  regression with crossed random effects. *Electronic Journal of
  Statistics* 16(2), 4604–4635.
