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

In `glmnet.gam` the first is a method for choosing the penalty parameters,
which this page writes. It's a `SmoothingMethod`, around the library's
`moment_equations`. The second is `GAM`'s backfitting, and the third
`fixed_effects_`. None of the three forms the $N \times P$ design or a
$P \times P$ matrix.

```{code-cell} ipython3
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from glmnet.gam import GAM, GAMControl, re, linear, ExperimentalWarning

warnings.simplefilter('ignore', ExperimentalWarning)
```

## Writing the method

`GAM(method=...)` takes a string (`'fixed'`, `'REML'`, `'ML'`) or a
`SmoothingMethod`. A method chooses the penalty parameters of the terms
that `GAM` was not given: smoothing parameters, random effects' variances.
Ghosh, Hastie and Owen's first step is such a method.

**The moments.** For random intercepts of $K$ crossed factors, $r_i =
\sum_k a^k_{g_k(i)} + e_i$, Gao and Owen's statistics are linear in the
variances in expectation:

- the sum of squares within the levels of each factor;
- $N$ times the total sum of squares.

Their expectations' coefficients are sums of squared counts of the levels
and of their pairs. `glmnet.gam.moments.moment_equations(r, codes)` gives
the statistics and that matrix in $O(N)$, from the factors' level codes.
Solving the $(K + 1) \times (K + 1)$ system gives the variances and
$\sigma_E^2$.

**The method.** `select(gam, dispersion)` does the following:

1. It finds the terms whose parameters it is to choose
   (`estimated_terms(gam)`: the penalized terms given neither `lam` nor
   `df`), and checks that it can.
2. It fits the model without them (their smoothing parameters very large:
   their effects 0), by `gam.inner_fit()`, and takes the residuals.
3. It solves the moment equations, and sets each term's smoothing
   parameter to $\lambda_k = \hat\sigma_E^2/\hat\sigma_k^2$ (on the scale
   of the sum of the prior weights, `GAM`'s objective).
4. It reports what it did, as a `SmoothingSelection`.

`GAM` then fits with those parameters, by backfitting: the GLS fit and
the random effects' predictions.

```{code-cell} ipython3
from statsmodels.genmod.families import family as sm_family, links as sm_links

from glmnet.gam import SmoothingMethod
from glmnet.gam.selection import estimated_terms
from glmnet.gam.reml import SmoothingSelection
from glmnet.gam.terms import RandomEffectTerm
from glmnet.gam.moments import moment_equations


class CrossedMoments(SmoothingMethod):
    """Method-of-moments variances of crossed random intercepts (Gaussian, equal weights)."""
    name = 'moments'
    joint = False           # backfit: GAM never forms the design
    covariance_edf = False  # the terms' own edf, not mgcv's (the dense P x P covariance)

    def select(self, gam, dispersion=None):
        base = gam._family.base
        if not (isinstance(base, sm_family.Gaussian) and isinstance(base.link, sm_links.Identity)):
            raise ValueError('the method of moments is for the Gaussian family')
        weight = np.asarray(gam._weight, float)
        if not np.allclose(weight, weight[0]):
            raise ValueError('the method of moments needs equal prior weights')
        estimated = estimated_terms(gam)
        terms = gam.terms_
        for j in estimated:
            if not isinstance(terms[j], RandomEffectTerm) or terms[j].by is not None:
                raise ValueError(f'{terms[j].name}: give its lam (only random intercepts here)')

        # 1. the fit without the random effects, and its residuals
        for j in estimated:
            terms[j].lam = 1e12
        state = gam.inner_fit()
        r = np.asarray(gam._response, float) - np.asarray(state.mu, float)

        # 2. the moment equations, solved; a negative variance is set to 0
        U, A = moment_equations(r, [terms[j].codes_ for j in estimated])
        variances = np.linalg.solve(A, U)
        s2, phi = np.maximum(variances[:-1], 0.), float(variances[-1])

        # 3. the smoothing parameters, phi / s_k^2, on the prior weights' scale
        W = weight.sum()
        lam_raw = np.where(s2 > 0, phi / np.where(s2 > 0, s2, 1.), 1e12 * W)
        for j, lam in zip(estimated, lam_raw):
            terms[j].lam = float(lam) / W

        return SmoothingSelection(method=self.name, terms=[terms[j].name for j in estimated],
                                  lam=lam_raw / W, lam_raw=lam_raw, theta=np.log(lam_raw),
                                  dispersion=phi if dispersion is None else dispersion,
                                  score=np.nan, gradient=np.zeros(0),
                                  converged=bool(gam.converged_), n_evaluations=1)
```

### Why `select()` is designed this way

- **A method declares what the fit needs.** `GAM` has to decide some
  things before it fits, and the method states them as attributes:
  - `joint`: backfit, or build the design for the joint solve, which REML
    needs;
  - `covariance_edf`: whether to report mgcv's edf, which form the dense
    $P \times P$ covariance;
  - `experimental`: whether to warn.

  Saying no to the first two is what keeps this fit linear in $N$. A
  plain callback could not say it.
- **The method chooses; `GAM` fits.** `select()` sets the parameters on
  the terms and returns. Fitting, finalizing the terms, the dispersion and
  the covariances stay `GAM`'s, so a method can't get them wrong. Within
  `select()` a method may fit as often as it needs, with
  `gam.inner_fit()`. This one fits once, without the random effects.
  Another, such as mgcv's extended Fellner–Schall update, would alternate
  fits and updates.
- **A contract on the terms.** `estimated_terms(gam)` says which terms
  the method owns; terms given `lam` keep it, and those given `df` have
  it matched first. The method sets theirs with the terms' own setters
  (`lam`, `set_penalty_lams`, or `set_precision` for correlated random
  effects).
- **One report.** Whatever the method, `smoothing_` is a
  `SmoothingSelection`, and `variance_components_`, `dispersion_` and the
  covariances follow from the parameters it set. A user-given
  `dispersion` comes in, and the method's estimate goes back out.
- **REML is one method, not the framework.** `'REML'` and `'ML'` are the
  built-in `Laplace` method, which needs the Hessian operations of
  `glmnet.gam.linalg`. This method needs only residuals and counts.

The method is short, so this page writes it out instead of the library
providing it. It's specific to its setting (Gaussian, equal weights,
random intercepts), and it's a template for other methods.

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

- The method above is for a Gaussian response with equal prior weights,
  and random intercepts. Random slopes and non-Gaussian responses (Ghosh,
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
