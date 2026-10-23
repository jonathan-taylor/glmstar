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

# Gaussian processes as GAM terms

A Gaussian process regression (kriging) of $y$ on $x$ models the mean as

$$f(x) = \alpha_0 + \alpha^\top x + g(x), \qquad g \sim \mathrm{GP}\big(0, \tau^2 C(\cdot, \cdot)\big),$$

a linear trend plus a zero mean Gaussian process with correlation function
$C$. Its posterior mean given the data is a penalized fit: $g$ is a
combination $\sum_i \beta_i C(x, x_i)$ of the correlation function centered
at the data, penalized by $\beta^\top C \beta$, the squared norm of $g$ in
the reproducing kernel Hilbert space of $C$ (Rasmussen and Williams, 2006,
chapter 6). That is a smooth term of a GAM, with one smoothing parameter,
the ratio of the noise variance to $\tau^2$: Kammann and Wand (2003) made
it one, with a low rank basis, in their *geoadditive* models, and mgcv
implements it as `s(x, bs="gp")` (Wood, 2017, section 5.8.2).

`glmnet.gam.gp` is that term, constructed as mgcv's:

- the knots $x_i$ are the distinct values (rows) of its columns, or a
  random 2000 of them when there are more (`max_knots`), or `knots=`;
- the rank is reduced (Wood, 2003): with $C(K, K) = U \Lambda U^\top$ at the
  knots, it keeps the eigenvectors of the $k - M$ largest eigenvalues, and
  its basis is $[C(x, K)\, U \Lambda^{-1/2},\ t(x)]$, the penalty the
  identity on the first block, $t(x) = (1, x)$ the trend ($M = d + 1$ in $d$
  dimensions) or $t(x) = 1$ with `trend=False` (a stationary process; mgcv's
  negative `m[1]`). The default `k` is mgcv's: $M + 10$, $M + 30$ and
  $M + 100$ in 1, 2 and more dimensions;
- the correlation functions are mgcv's, of the scaled distance
  $d = \|x - x'\| / \rho$: the Matern with smoothness $\nu = 1.5$ (the
  default), 2.5 or 3.5, e.g. $e^{-d}(1 + d)$ for $\nu = 1.5$; the power
  exponential $e^{-d^p}$, $0 < p \le 2$ (`kernel='exponential'`; $p = 2$ is
  the squared exponential); and the spherical. The range $\rho$ defaults to
  the largest distance between the knots (Kammann and Wand's choice).

Like a spline term `s()`, it takes a fixed smoothing parameter `lam`, a
number of degrees of freedom `df`, or neither, to have its smoothing
parameter estimated by REML or ML; it works with both solvers (backfitting
and the joint fit), every family, weights and offsets, and has standard
errors and `plot`.

```{code-cell} ipython3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from glmnet.gam import GAM, GLMM, gaussian_process, s, re, categorical, plot_term

rng = np.random.default_rng(0)
```

## A curve

A function with a sharp feature, observed with noise at 200 points:

```{code-cell} ipython3
def f_true(x):
    return np.sin(6 * x) + 1.5 * np.exp(-((x - 0.7) / 0.05)**2)

n = 200
curve = pd.DataFrame({'x': np.sort(rng.uniform(0, 1, n))})
curve['y'] = f_true(curve['x']) + 0.3 * rng.standard_normal(n)
```

A GP term and a cubic smoothing spline, each with its smoothing parameter
chosen by REML:

```{code-cell} ipython3
fit_gp = GLMM(terms=[gaussian_process('x', k=30)], method='REML').fit(curve, curve['y'])
fit_s = GLMM(terms=[s('x')], method='REML').fit(curve, curve['y'])
pd.DataFrame({'edf': [fit_gp.edf_.iloc[0], fit_s.edf_.iloc[0]],
              'REML': [fit_gp.reml_score_, fit_s.reml_score_]},
             index=['gp(x, k=30)', 's(x)'])
```

```{code-cell} ipython3
grid = pd.DataFrame({'x': np.linspace(0, 1, 300)})
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve['x'], curve['y'], '.', color='0.6', label='data')
ax.plot(grid['x'], f_true(grid['x']), 'k:', label='truth')
for fit, label in [(fit_gp, 'gp(x), Matern 1.5'), (fit_s, 's(x), cubic spline')]:
    mean, se = fit.predict(grid, return_std=True)
    line, = ax.plot(grid['x'], mean, label=label)
    ax.fill_between(grid['x'], mean - 2 * se, mean + 2 * se, color=line.get_color(), alpha=0.15)
ax.legend();
```

The two are close (the GP's curve is mostly hidden under the spline's): a
cubic smoothing spline is itself the posterior mean of
a (partially improper) Gaussian process (Wahba, 1990). The Matern with
$\nu = 1.5$ has sample paths with one derivative, so it can bend more
sharply at the bump.

## The kernel and its range

The range sets how far correlations reach: a short range lets the fit
bend over short distances, a long one makes the process nearly as
smooth as its trend, with the smoothing parameter estimated in each case.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve['x'], curve['y'], '.', color='0.7')
for rho in [0.02, 0.1, 0.5, 2.]:
    fit = GLMM(terms=[gaussian_process('x', k=30, range=rho)], method='REML').fit(curve, curve['y'])
    ax.plot(grid['x'], fit.predict(grid), label=f'range {rho}: edf {fit.edf_.iloc[0]:.1f}')
ax.legend();
```

The range is not estimated with the smoothing parameter, but the REML
criterion of each fit is the (restricted) marginal likelihood of the data
given the range, so it can be profiled over a grid of ranges and
minimized, as Wood (2017, section 5.8.2) does. The same goes for the choice
of kernel:

```{code-cell} ipython3
ranges = np.geomspace(0.01, 0.5, 15)
kernels = {'Matern 1.5': dict(nu=1.5), 'Matern 2.5': dict(nu=2.5),
           'Matern 3.5': dict(nu=3.5), 'exponential': dict(nu=0.5)}
t0 = time.time()
profile = pd.DataFrame({name: [GLMM(terms=[gaussian_process('x', k=20, range=rho, **kw)],
                                   method='REML')
                               .fit(curve, curve['y']).reml_score_ for rho in ranges]
                        for name, kw in kernels.items()}, index=ranges)
print(f'{profile.size} fits in {time.time() - t0:.1f}s')
ax = profile.plot(logx=True, figsize=(7, 4))
ax.set_xlabel('range'); ax.set_ylabel('REML criterion (negative log)');
```

```{code-cell} ipython3
best = profile.stack().idxmin()
print(f'smallest criterion: {best[1]}, range {best[0]:.3f}')
```

(The criteria of different ranges and kernels are comparable: the term's
unpenalized part, the trend, is the same, and the penalty enters with its
normalizing constant. The smoother kernels, at long ranges, are nearly of
low rank: the eigenvalues of their correlation matrix fall off fast, and
those below $10^{-12}$ of the largest, rounding, are left out of the basis,
with a warning; hence the grid stops at 0.5 here.)

## A surface

In two (or more) columns `gaussian_process` is an isotropic process: distances are
Euclidean, so the columns should be on comparable scales (coordinates in
the same units). Simulated data at 500 irregular locations:

```{code-cell} ipython3
def surface(u, v):
    return (np.exp(-((u - 0.3)**2 + (v - 0.7)**2) / 0.03) -
            0.8 * np.exp(-((u - 0.7)**2 + (v - 0.3)**2) / 0.05) + 0.5 * u)

n = 500
spatial = pd.DataFrame({'u': rng.uniform(0, 1, n), 'v': rng.uniform(0, 1, n)})
spatial['y'] = surface(spatial['u'], spatial['v']) + 0.2 * rng.standard_normal(n)
t0 = time.time()
fit_2d = GLMM(terms=[gaussian_process('u', 'v', k=60)], method='REML').fit(spatial, spatial['y'])
print(f'{time.time() - t0:.1f}s')
fit_2d.summary()
```

The model, as `GAM` displays it:

```{code-cell} ipython3
fit_2d
```

`plot` draws a two column term as filled contours over the data's bounding
box (leaving out the parts far from any data, as mgcv's `too.far`), with
the locations as points:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_term(fit_2d, 0, ax=axes[0])
U, V = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
truth = surface(U, V)
cs = axes[1].contourf(U, V, truth - truth.mean(), levels=15)
fig.colorbar(cs, ax=axes[1])
axes[1].set_title('truth (centered)')
fig.tight_layout()
```

The standard errors of the surface, from the posterior covariance, are
those of `partial_effect`; they grow away from the data:

```{code-cell} ipython3
pts = pd.DataFrame({'u': [0.3, 0.5, 0.98], 'v': [0.7, 0.5, 0.02]})
fit, se = fit_2d.partial_effect(0, pts, return_std=True)
pd.DataFrame({'u': pts['u'], 'v': pts['v'], 'fit': fit, 'truth': surface(pts['u'], pts['v']) - truth.mean(),
              'std err': se})
```

## A binary response

Any family: presence or absence, with the log odds a surface:

```{code-cell} ipython3
n = 800
pres = pd.DataFrame({'u': rng.uniform(0, 1, n), 'v': rng.uniform(0, 1, n)})
pres['y'] = rng.binomial(1, 1 / (1 + np.exp(-2 * surface(pres['u'], pres['v']))))
fit_b = GLMM(terms=[gaussian_process('u', 'v', k=40)], family=sm.families.Binomial(),
            method='REML').fit(pres, pres['y'])
print(f"edf {fit_b.edf_.iloc[0]:.1f}")
P = fit_b.predict(pd.DataFrame({'u': U.ravel(), 'v': V.ravel()})).reshape(U.shape)
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(U, V, P, levels=np.linspace(0, 1, 11), cmap='RdBu_r')
fig.colorbar(cs, ax=ax, label='P(y = 1)')
ax.plot(pres['u'][pres['y'] == 1], pres['v'][pres['y'] == 1], 'k.', ms=2);
```

## A process and a random effect

Measurements repeated at 60 sites, with an effect of each site beyond the
spatial trend (a *nugget* that is shared by the site's measurements): a
GP term in the coordinates plus a random intercept per site, both
estimated by REML. The sites are the knots.

```{code-cell} ipython3
sites = pd.DataFrame({'site': [f's{i:02d}' for i in range(60)],
                      'u': rng.uniform(0, 1, 60), 'v': rng.uniform(0, 1, 60),
                      'effect': 0.3 * rng.standard_normal(60)})
obs = sites.loc[np.repeat(np.arange(60), 8)].reset_index(drop=True)
obs['y'] = surface(obs['u'], obs['v']) + obs['effect'] + 0.2 * rng.standard_normal(obs.shape[0])
fit_re = GLMM(terms=[gaussian_process('u', 'v'), re('site')], method='REML').fit(obs, obs['y'])
fit_re.variance_components_
```

A Gaussian process surface with a random effect of the sites:

```{code-cell} ipython3
fit_re
```

The site effects' standard deviation (0.3 in the simulation) and the
residual one (0.2) are estimated with the surface. The term's variance is
that of the coefficients of its basis (the process' $\tau^2$ in the
normalized basis).

## Against scikit-learn's Gaussian process regression

With a knot at every distinct value (`k` the number of them plus 1), no
trend (`trend=False`), and the range and smoothing parameter fixed, the
Gaussian GAM is scikit-learn's `GaussianProcessRegressor` at fixed
hyperparameters: its posterior mean is ours when the noise variance
$\sigma^2$ and the process variance $\tau^2$ have
$\sigma^2 / \tau^2 = \lambda \sum_i w_i$ (the GAM's objective is the deviance
over twice the sum of the weights), with a near flat prior on the
intercept (a constant kernel of large variance) and scikit-learn's Matern
of length scale $\sqrt{2\nu}\,\rho$. The posterior standard deviations agree
at the knots (elsewhere scikit-learn's adds the prior variance of the part
of the process orthogonal to the knots, which the basis leaves out), and
our REML criterion is minus scikit-learn's log marginal likelihood, up to
the constant of its intercept's prior (`tests/gam/test_gam_gp.py`).

```{code-cell} ipython3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

sigma2, lam, rho = 0.09, 1e-3, 0.05
nk = curve['x'].nunique()
ours = GAM(terms=[gaussian_process('x', range=rho, trend=False, k=nk + 1, lam=lam)])
ours = ours.fit(curve, curve['y'], dispersion=sigma2)
tau2 = sigma2 / (lam * curve.shape[0])
kernel = (ConstantKernel(tau2, 'fixed') * Matern(np.sqrt(3) * rho, nu=1.5, length_scale_bounds='fixed')
          + ConstantKernel(1e5, 'fixed'))
skl = GaussianProcessRegressor(kernel, alpha=sigma2, optimizer=None).fit(curve[['x']], curve['y'])
mean, sd = ours.predict(curve, return_std=True)
mean_R, sd_R = skl.predict(curve[['x']], return_std=True)
print(f'largest differences: mean {np.abs(mean - mean_R).max():.1e}, sd {np.abs(sd - sd_R).max():.1e}')
```

## Notes

- **mgcv.** With the same knots, `k` and `m`, the term spans mgcv's basis
  and has its penalty: at fixed smoothing parameters (mgcv's `sp` is
  `lam * sum(weights) * S.scale`) the fits, the edf and the standard
  errors agree, and REML and ML estimate the same smoothing parameters
  (`tests/gam/test_gam_gp_r.py`). mgcv's `m = c(code, range, power)` can be
  given as `gaussian_process(..., m=(code, range, power))`. mgcv finds the eigenvectors
  by Lanczos iterations to a tolerance of about $10^{-8}$ (relative to the
  largest eigenvalue), so that its eigenvectors of small eigenvalues, and
  its fits, are accurate to about $10^{-6}$ to $10^{-4}$; ours are exact.
  Where there are more than `max_knots` distinct rows, both draw knots at
  random, but not the same ones; pass `knots=` to fix them.
- **Cost.** The basis needs the eigendecomposition of the knots'
  correlation matrix (up to 2000 by 2000, a few seconds) once, and is an
  $n$ by $k$ matrix; fits cost as those of any term with that many
  coefficients.
- **The range** is fixed for each fit: profile it, as above, or set it from
  knowledge of the scale of the spatial dependence.

## References

- Kammann, E. E. and Wand, M. P. (2003). Geoadditive models. *Journal of
  the Royal Statistical Society, Series C* 52(1), 1–18.
- Handcock, M. S., Meier, K. and Nychka, D. (1994). Comment on "Kriging and
  splines: an empirical comparison of their predictive performance in some
  applications". *Journal of the American Statistical Association* 89(426),
  401–403.
- Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
- Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
  Statistical Society, Series B* 65(1), 95–114.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
