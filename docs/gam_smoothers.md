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

# A gallery of smoothers

Each term of a GAM is a smoother ([Generalized additive
models](gam.md#smoothers)), and each smoother has a parameter that sets how
much it smooths: a spline's degrees of freedom or smoothing parameter, a
loess's span, a basis' size, a Gaussian process' range, a tree's depth.
This page fits a GAM of one term to the same simulated data for several
values of that parameter, and draws each fit's `predict()` on a grid.

The parameters are a smoother's arguments, so they are also what
`GridSearchCV` tunes (`terms__0__df`, ...; [Choosing a smoother by
cross-validation](gam.md#choosing-a-smoother-by-cross-validation)), and
what a `GLMM` estimates when a smoothing parameter is left out
([Generalized linear mixed models](glmm.md)).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glmnet.gam import (GAM, GAMControl, s, lo, cubic_regression_spline, cyclic_spline, gaussian_process,
                        tensor_spline, random_effect as re, categorical, TransformerSmoother, RegressorSmoother)

rng = np.random.default_rng(0)
n = 300
x = np.sort(rng.uniform(0, 1, n))
f = np.sin(8 * x) * np.exp(-x) + (x > 0.7) * 0.5
D = pd.DataFrame({'x': x, 'y': f + 0.25 * rng.standard_normal(n)})
grid = pd.DataFrame({'x': np.linspace(0, 1, 400)})


def gallery(make, values, label, data=D, ax=None, truth=f):
    """Fit GAM([make(v)]) for each value; draw the data, the truth and each fit's predict()."""
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    ax.scatter(data['x'], data['y'], s=6, color='0.75')
    ax.plot(data['x'], truth, color='k', lw=1, ls='--', label='truth')
    for v in values:
        fit = GAM(terms=[make(v)]).fit(data, data['y'])
        ax.plot(grid['x'], fit.predict(grid), lw=2, label=f'{label} = {v}')
    ax.legend(fontsize=8)
    return ax
```

## Smoothing splines: `s(x, df=...)`

The natural cubic smoothing spline, knots at the data (as R's
`smooth.spline`). `df` is its degrees of freedom (the trace of its
smoother, less the constant), matched at the working weights; `lam` would
give the smoothing parameter directly. Few degrees of freedom: nearly a
line; many: it follows the noise.

```{code-cell} ipython3
ax = gallery(lambda v: s('x', df=v), [2, 4, 8, 20], 'df')
ax.set_title("s('x', df=...)");
```

## Loess: `lo(x, span=...)`

Local regression (Cleveland, Grosse and Shyu, 1992): each fitted value a
weighted least squares fit on the `span` fraction of the data nearest it,
of degree 1 (or 2). It has no basis: it is fitted by backfitting, and has
no standard errors.

```{code-cell} ipython3
ax = gallery(lambda v: lo('x', span=v), [0.1, 0.3, 0.75], 'span')
ax.set_title("lo('x', span=...)");
```

## Regression splines: `cubic_regression_spline(x, k=...)`

mgcv's `bs="cr"`: a natural cubic spline with `k` knots at quantiles of
the data, penalized by $\int f''^2$. Here the smoothing parameter is small
and fixed, so `k`, the basis' size, does the smoothing: a few knots cannot
follow the jump.

```{code-cell} ipython3
ax = gallery(lambda v: cubic_regression_spline('x', k=v, lam=1e-7), [4, 6, 10, 25], 'k')
ax.set_title("cubic_regression_spline('x', k=..., lam=1e-7)");
```

and with `k` fixed, `lam` does:

```{code-cell} ipython3
ax = gallery(lambda v: cubic_regression_spline('x', k=25, lam=v), [1e-7, 1e-4, 1e-2], 'lam')
ax.set_title("cubic_regression_spline('x', k=25, lam=...)");
```

## Cyclic splines: `cyclic_spline(x, period=..., lam=...)`

A cubic spline whose value, slope and curvature agree at the ends of the
period: for the hour of the day, the day of the year. On a periodic
function, the larger `lam` the closer to a constant (its penalty's null
space: a periodic function's only line is flat):

```{code-cell} ipython3
fc = np.sin(2 * np.pi * x) + 0.5 * np.cos(6 * np.pi * x)
Dc = pd.DataFrame({'x': x, 'y': fc + 0.3 * rng.standard_normal(n)})
ax = gallery(lambda v: cyclic_spline('x', period=(0, 1), k=12, lam=v), [1e-7, 1e-5, 1e-3],
             'lam', data=Dc, truth=fc)
ax.set_title("cyclic_spline('x', period=(0, 1), lam=...)");
```

## Gaussian processes: `gaussian_process(x, range=...)`

Low rank kriging (Kammann and Wand, 2003; mgcv's `bs="gp"`): a Matérn
correlation of the distances over the `range`, and a linear trend. A short
range lets the fit bend quickly, a long one makes it nearly the trend;
`lam` (fixed here) weighs the process against the data.

```{code-cell} ipython3
ax = gallery(lambda v: gaussian_process('x', range=v, k=40, lam=1e-3), [0.02, 0.1, 0.5, 2.],
             'range')
ax.set_title("gaussian_process('x', range=..., lam=1e-3)");
```

## A P-spline from scikit-learn: `TransformerSmoother`

`SplineTransformer`'s B-splines with a penalty on their coefficients'
second differences (Eilers and Marx, 1996). The number of knots matters
little once there are enough; the penalty does the smoothing:

```{code-cell} ipython3
from sklearn.preprocessing import SplineTransformer

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
gallery(lambda v: TransformerSmoother('x', SplineTransformer(n_knots=v), lam=1e-4),
        [5, 10, 40], 'n_knots', ax=axes[0])
axes[0].set_title('SplineTransformer(n_knots=...), lam=1e-4')
gallery(lambda v: TransformerSmoother('x', SplineTransformer(n_knots=40), lam=v),
        [1e-6, 1e-3, 1e-1], 'lam', ax=axes[1])
axes[1].set_title('SplineTransformer(n_knots=40), lam=...')
fig.tight_layout()
```

## A tree: `RegressorSmoother`

Any scikit-learn regressor that takes `sample_weight` is a smoother for
backfitting. A tree's depth sets its number of steps; the jump at 0.7 is
easy for it, the waves are not:

```{code-cell} ipython3
from sklearn.tree import DecisionTreeRegressor

ax = gallery(lambda v: RegressorSmoother('x', DecisionTreeRegressor(max_depth=v), df=2 ** v - 1),
             [1, 2, 3, 6], 'max_depth')
ax.set_title("RegressorSmoother('x', DecisionTreeRegressor(max_depth=...))");
```

(`df` is given, its number of leaves less 1, to skip the Monte Carlo estimate of
its generalized degrees of freedom; it changes only the fit's degrees of
freedom, not its predictions.)

## Random effects: `random_effect(g, lam=...)`

A random effect's prediction for a group is its effect: the group's mean
residual shrunk towards 0, by $n_g / (n_g + \lambda \sum w)$ for a group
of $n_g$ observations (with unit weights). Small groups are shrunk the
most. With `lam` 0 the effects are the groups' means, as `categorical()`'s:

```{code-cell} ipython3
G = 12
sizes = rng.integers(2, 40, G)
g = np.repeat(np.arange(G), sizes)
effects = rng.normal(0, 1, G)
Dg = pd.DataFrame({'g': g, 'y': effects[g] + rng.standard_normal(len(g))})
levels = pd.DataFrame({'g': np.arange(G)})

fig, ax = plt.subplots(figsize=(6, 3.5))
means = Dg.groupby('g')['y'].mean() - Dg['y'].mean()
order = np.argsort(sizes)
ax.plot(sizes[order], means.to_numpy()[order], 'ko', label='group means')
for lam in [1e-3, 1e-2, 1e-1]:
    # (jointly: with a small lam the intercept and the effects are nearly
    # confounded, and backfitting between them is slow)
    fit = GAM(terms=[re('g', lam=lam)], control=GAMControl(solver='joint')).fit(Dg, Dg['y'])
    ax.plot(sizes[order], (fit.predict(levels) - fit.intercept_)[order], 'o', ms=4,
            label=f'lam = {lam}')
ax.axhline(0, color='0.5', lw=0.5)
ax.set_xlabel('observations in the group'); ax.set_ylabel('predicted effect')
ax.legend(fontsize=8); ax.set_title("random_effect('g', lam=...)");
```

## Tensor products: `tensor_spline(x, z, lam=[...])`

A smooth of two columns with a smoothing parameter for each: a large one
makes the surface linear along that column (its penalty's null space),
whatever it does along the other.

```{code-cell} ipython3
x2, z2 = rng.uniform(0, 1, (2, 600))
D2 = pd.DataFrame({'x': x2, 'z': z2,
                   'y': np.sin(6 * x2) * np.cos(4 * z2) + 0.2 * rng.standard_normal(600)})
xs = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, lam in zip(axes, [[1e-4, 1e-4], [1e2, 1e-4], [1e-4, 1e2]]):
    fit = GAM(terms=[tensor_spline('x', 'z', k=(8, 8), lam=lam)]).fit(D2, D2['y'])
    for z0 in [0.1, 0.5, 0.9]:
        ax.plot(xs, fit.predict(pd.DataFrame({'x': xs, 'z': z0})), label=f'z = {z0}')
    ax.set_title(f'lam = {lam}'); ax.set_xlabel('x')
axes[0].set_ylabel('predict(x, z)'); axes[0].legend(fontsize=8)
fig.tight_layout()
```

Each panel draws the surface along `x` at three values of `z`. With the
first smoothing parameter large (middle), the surface is linear in `x` at
each `z`, its slope changing with `z`. With the second large (right), it is
linear in `z`: the three slices are then equally spaced.

## Without a parameter

`linear(x)` (a line) and `categorical(g)` (an effect per level) have none:
they are the penalties' null spaces of the others, fitted without
shrinkage. `varying(term, by=...)` takes its term's parameters, per level
of `by` or shared (`pooled_lam=True`).

## References

- Cleveland, W. S., Grosse, E. and Shyu, W. M. (1992). Local regression
  models. In J. M. Chambers and T. J. Hastie (eds.), *Statistical Models in
  S*, chapter 8. Wadsworth & Brooks/Cole.
- Eilers, P. H. C. and Marx, B. D. (1996). Flexible smoothing with
  B-splines and penalties. *Statistical Science* 11(2), 89–121.
- Kammann, E. E. and Wand, M. P. (2003). Geoadditive models. *Journal of
  the Royal Statistical Society, Series C* 52(1), 1–18.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
