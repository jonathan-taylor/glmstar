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

# Overdispersed Poisson regression

Counts are often more variable than the Poisson model allows. A common
remedy is the quasi-Poisson model ({ref}`Wedderburn, 1974 <od-poisson-refs>`):
the mean is still log-linear, $\log \mu_i = \beta_0 + x_i^T\beta$, but the
variance is $\mathrm{Var}(y_i) = \phi\,\mu_i$ with a dispersion
$\phi > 1$. The quasi-likelihood score is the Poisson score, so the
Poisson lasso fit by `GLMNet` is also the quasi-Poisson lasso. Only the
variances change: the score has $\phi$ times its Poisson variance.

This vignette repeats the inference after selection of
[Score bootstraps with a linear term](score_bootstrap.md) for an
overdispersed Poisson regression. As there, we select with the lasso
fit to the counts with a random linear term $-\omega$, which perturbs
the score, and then either combine two fits or carve with `lassoinf`. The randomization and the inference both
have to account for $\phi$. We show what happens when they do, and when
they do not.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl
from glmnet.inference import glmstar_inference_problem, score_perturbation
from lassoinf import LassoInference

from experiments import load_results, provenance
```

## Data

We simulate counts with $\mathrm{Var}(y_i) = \phi\mu_i$ and $\phi = 3$,
using a negative binomial distribution with size $\mu_i/(\phi - 1)$.
As in the logistic example, there are two strong signals, six weak ones
near the selection threshold, and 22 nulls. The features are correlated
and have standard deviations from 0.2 to 5, and the effects are set per
standard deviation of each feature.

```{code-cell} ipython3
rng = np.random.default_rng(2)
n, p, rho, phi = 5000, 30, 0.3, 3.0
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.3, -0.25, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def sim_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    mu = np.exp(0.5 + X @ beta)
    size = mu / (phi - 1)          # negative binomial with variance phi * mu
    return X, rng.negative_binomial(size, size / (size + mu)).astype(float)

X, y = sim_data(n)
```

## Estimating the dispersion

We estimate $\phi$ from the unpenalized Poisson regression on all $p$
features, by Pearson's statistic

$$\hat\phi = \frac{1}{n - p - 1}\sum_{i=1}^n \frac{(y_i - \hat\mu_i)^2}{\hat\mu_i}.$$

```{code-cell} ipython3
def poisson_fit(X, y):
    return sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit()

def dispersion(X, y):
    fit = poisson_fit(X, y)
    phi_hat = (fit.resid_pearson**2).sum() / (len(y) - X.shape[1] - 1)
    return phi_hat, np.log(fit.fittedvalues)

phi_hat, eta_hat = dispersion(X, y)
phi_hat
```

## The randomized lasso

`poisson_lasso` fits the lasso at one value of $\lambda$, with an
optional linear term `LinearTerm(coef=G, intercept=G_0)` adding
$G_0\beta_0 + G^T\beta$ to the objective. The counts themselves are
never changed.

```{code-cell} ipython3
pi = 0.9
lam = 2 * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def poisson_lasso(X, y, linear_term=None, relax=False, control=None):
    return GLMNet(family=sm.families.Poisson(),
                  lambda_values=np.array([lam]),
                  linear_term=linear_term,
                  relax=relax,
                  control=control or GLMNetControl()).fit(X, y)
```

## A perturbation for overdispersed counts

Draw two independent Poisson replicates
$\check y_1, \check y_2 \sim \mathrm{Poisson}(\hat\mu)$ from the fitted
means. As for logistic regression, $(\check y_1 - \check y_2)/\sqrt 2$
has mean 0 and the Poisson variance $\hat\mu_i$. The data have
$\phi$ times this variance, so we scale it by $\sqrt{\hat\phi}$:

$$e = \sqrt{\hat\phi}\;\sqrt{\frac{1-\pi}{\pi}}\;\frac{\check y_1 - \check y_2}{\sqrt 2}, \qquad \mathrm{Var}(e_i \mid \hat\mu) = \hat\phi\,\frac{1-\pi}{\pi}\,\hat\mu_i.$$

The score perturbation $\omega = \sum_i v_i e_i\,(1, x_i)$, with
$v_i = 1/n$, then has $(1-\pi)/\pi$ times the covariance of the score,
$\hat\phi\,X_1^TV\,\mathrm{diag}(\hat\mu)\,VX_1$ with
$X_1 = [\mathbf{1}, X]$ and $V = \mathrm{diag}(v)$. This is the relation
that makes selection with the linear term $-\omega$ imitate selection
on a random fraction $\pi$ of the data. Without the factor
$\sqrt{\hat\phi}$, the perturbation is too small by a factor of $\phi$
in variance. `score_perturbation` draws $-\omega$ from the pilot fit
$\hat\eta = \log\hat\mu$, with `dispersion` the factor $\hat\phi$:

```{code-cell} ipython3
def perturbation(X, eta_hat, phi_hat):
    return score_perturbation(X, eta_hat, sm.families.Poisson(), (1 - pi) / pi,
                              dispersion=phi_hat, rng=rng)
```

## Inference with the dispersion

Both ways of doing inference from the logistic example carry over, with
the dispersion in the variances.

- **Combination.** $T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$,
  where $\hat\beta_E$ is the Poisson fit of $y$ on $X_E$ and
  $\hat\beta^*_E$ the relaxed fit of the randomized lasso. Its
  standard errors are the Poisson ones times $\sqrt{\hat\phi/(1-\pi)}$.
- **Carving.** The selection problem is the lasso problem for $y$ with
  the linear term $-\omega$. `glmstar_inference_problem` gets
  `dispersion=phi_hat`, which multiplies the covariance of the score,
  and the same `scalar_noise = (1 - pi) / pi`.

`randomized_inference` does one randomized selection and returns both
sets of intervals. With `phi_hat=1` it ignores the overdispersion, in
the perturbation and in the inference.

```{code-cell} ipython3
def randomized_inference(X, y, eta_hat, phi_hat):
    omega = perturbation(X, eta_hat, phi_hat)
    fit = poisson_lasso(X, y, linear_term=omega, relax=True, control=tight)
    E = np.nonzero(fit.coefs_[0])[0]
    if len(E) == 0:
        return E, None

    # combination
    full = poisson_fit(X[:, E], y)
    T = (full.params[1:] - pi * fit.relaxed_coefs_[0][E]) / (1 - pi)
    se = full.bse[1:] * np.sqrt(phi_hat / (1 - pi))
    combination = (T, T - q * se, T + q * se)

    # carving
    info = glmstar_inference_problem(fit, X, y, dispersion=phi_hat, scalar_noise=(1 - pi) / pi)
    # row 0 is the intercept
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_.loc[E + 1]
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the Poisson regression of $y$ on
$X_E$ in the population, which the quasi-Poisson model shares. We
approximate them by fitting each selected model to one large sample:

```{code-cell} ipython3
pop_X, pop_y = sim_data(200000)
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        targets[key] = poisson_fit(pop_X[:, E], pop_y).params[1:]
    return targets[key]
```

### One data set

```{code-cell} ipython3
E, intervals = randomized_inference(X, y, eta_hat, phi_hat)
table = pd.concat({m: pd.DataFrame({'estimate': est, 'lower': lower, 'upper': upper},
                                   index=[f'X{j}' for j in E])
                   for m, (est, lower, upper) in intervals.items()}, axis=1)
table['target'] = target(E)
table.round(3)
```

## What ignoring the overdispersion does

Suppose we draw $e$ without the factor $\sqrt{\hat\phi}$ and use
$\phi = 1$ in the inference. Write $\Sigma$ for the Poisson covariance
of the score $S$, so that $\mathrm{Var}(S) = \phi\Sigma$, while
$\mathrm{Var}(\omega) = \frac{1-\pi}{\pi}\Sigma$.

- **Combination.** $T_E$ was independent of the selection because
  $S - \frac{\pi}{1-\pi}\omega$ is uncorrelated with the selection score
  $S + \omega$. Now
  $\mathrm{Cov}\big(S + \omega, S - \tfrac{\pi}{1-\pi}\omega\big) = \phi\Sigma - \Sigma = (\phi - 1)\Sigma$,
  so $T_E$ still depends on the selection. Its standard errors are also
  too small.
- **Carving.** `lassoinf` takes the score's covariance to be $\Sigma$
  when it is $\phi\Sigma$, and the randomization to be $(1-\pi)/\pi$
  times the score's variance when it is $(1-\pi)/(\pi\phi)$ times it.
  The intervals are too short, and the conditioning on the selection is
  wrong.

## Calibration

We repeat the experiment and compare six methods:

- **naive**: select with the lasso on the data, then use the
  full-data intervals, with $\hat\phi$. This accounts for the
  overdispersion but not for the selection;
- **split**: select on a random 90% of the data, then form intervals on
  the other 10%, with $\phi$ estimated on that 10%;
- **combination** and **carved**: as above, with $\hat\phi$;
- **combination ($\phi = 1$)** and **carved ($\phi = 1$)**: the same,
  ignoring the overdispersion.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length per standard
deviation of the feature.

```{code-cell} ipython3
def calibration(n_rep=300):
    rows = []

    def record(method, E, lower, upper):
        for j, lo, hi, t in zip(E, lower, upper, target(E)):
            rows.append({'method': method, 'signal': signal[j],
                         'covered': lo <= t <= hi, 'length': (hi - lo) * scale[j]})

    def poisson_intervals(X, y, E, phi_hat):
        fit = poisson_fit(X[:, E], y)
        se = fit.bse[1:] * np.sqrt(phi_hat)
        return fit.params[1:] - q * se, fit.params[1:] + q * se

    for _ in range(n_rep):
        X, y = sim_data(n)
        phi_hat, eta_hat = dispersion(X, y)

        E = np.nonzero(poisson_lasso(X, y).coefs_[0])[0]
        if len(E):
            record('naive', E, *poisson_intervals(X, y, E, phi_hat))

        idx = rng.permutation(n)
        sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
        E = np.nonzero(poisson_lasso(X[sel], y[sel]).coefs_[0])[0]
        if len(E):
            phi_held_out = dispersion(X[held_out], y[held_out])[0]
            record('split', E, *poisson_intervals(X[held_out], y[held_out], E, phi_held_out))

        for suffix, phi_used in [('', phi_hat), (' (phi = 1)', 1.)]:
            E, intervals = randomized_inference(X, y, eta_hat, phi_used)
            if len(E):
                for method, (_, lower, upper) in intervals.items():
                    record(method + suffix, E, lower, upper)

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py overdispersed_poisson`
and saved in `docs/results/overdispersed_poisson.csv`.

```{code-cell} ipython3
results = load_results('overdispersed_poisson')
display(provenance(results))

methods = ['naive', 'split', 'combination', 'carved',
           'combination (phi = 1)', 'carved (phi = 1)']
results['method'] = pd.Categorical(results['method'], methods)
results['signal'] = pd.Categorical(results['signal'], ['strong', 'weak', 'null'])
results.groupby('method', observed=True).agg(intervals=('covered', 'size'),
                                             coverage=('covered', 'mean'),
                                             mean_length=('length', 'mean')).round(3)
```

By the strength of the signal:

```{code-cell} ipython3
(results.groupby(['signal', 'method'], observed=True)
        .agg(intervals=('covered', 'size'),
             coverage=('covered', 'mean'),
             mean_length=('length', 'mean'))
        .round(3))
```

With $\hat\phi$, the split, combination and carved intervals are
calibrated. As in the logistic example, the carved intervals are the
shortest of the three. For the strong signals they are close to the
naive intervals, which are fine there because those variables would be
selected anyway. The naive intervals fail for the selected nulls.

Ignoring the overdispersion breaks the randomized methods. The carved
intervals become too short by about $\sqrt\phi$ and undercover for
every kind of signal. The combination intervals undercover less for
the strong signals, whose selection is nearly certain. For the nulls,
the leftover correlation $(\phi - 1)\Sigma$ with the selection matters,
and they undercover badly.

The perturbation relies on the quasi-Poisson model: a constant $\phi$
with $\mathrm{Var}(y_i) = \phi\mu_i$. If the variance has another
form, such as the negative binomial's $\mu + \mu^2/k$ with a fixed
size $k$, the scaled Poisson replicates no longer match the variance
of the score. The randomization can then come from a nonparametric
resample of the observations instead
({ref}`Fry, Panigrahi and Taylor, 2025 <od-poisson-refs>`).

(od-poisson-refs)=
## References

- Wedderburn, R. W. M. (1974). Quasi-likelihood functions, generalized
  linear models, and the Gauss–Newton method. *Biometrika*, 61(3),
  439–447.
- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
