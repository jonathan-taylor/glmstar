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

# Autocorrelated errors

The vignettes so far assume independent observations. Here the
observations form a time series, and the errors of the linear model

$$y_t = \beta_0 + x_t^T\beta + \epsilon_t, \qquad t = 1, \dots, n,$$

follow a stationary AR(1) process,
$\epsilon_t = \rho\,\epsilon_{t-1} + \sigma\eta_t$ with
$\eta_t \sim N(0, 1)$. The lasso is fit by `GLMNet` as usual, by least
squares, but the score

$$S = \frac1n X_1^T(y - X_1\beta^\circ), \qquad X_1 = [\mathbf 1, X],$$

no longer has the covariance $\sigma^2_\epsilon X_1^TX_1/n^2$ of
independent errors. It has $X_1^T\Sigma_\epsilon X_1/n^2$, with
$\Sigma_\epsilon$ the AR(1) covariance of the errors. When the features
are themselves autocorrelated, the two differ a lot.

We estimate $\rho$ and $\sigma$ from the residuals, and repeat the
inference after selection of
[Score bootstraps with a linear term](score_bootstrap.md): select with
the lasso fit to the data with a random linear term $-\omega$, then
either combine two fits or carve with `lassoinf`
({ref}`Tian and Taylor, 2018 <ar1-refs>`). The perturbation is
again the difference of two parametric draws, now from the fitted AR(1)
model, so it has the serial dependence of the errors.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from scipy.linalg import matmul_toeplitz
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.stats import norm

from glmnet import GLMNet, LinearTerm
from glmnet.glmnet import GLMNetControl
from glmnet.inference import glmstar_inference_problem
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
```

## Data

Each feature is an AR(1) series in time with coefficient
$\varphi = 0.8$, and the features are correlated with each other at
each time. The errors are AR(1) with $\rho = 0.7$. As in the other
examples there are two strong signals, six weak ones near the selection
threshold, and 22 nulls, the features have standard deviations from 0.2
to 5, and the effects are set per standard deviation of each feature.

```{code-cell} ipython3
rng = np.random.default_rng(4)
n, p, rho_x, phi, rho, sigma = 2000, 30, 0.3, 0.8, 0.7, 1.0
C = rho_x ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))   # correlation at each time
C_half = np.linalg.cholesky(C).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.6, -0.5, 0.2, -0.16, 0.13, 0.11, -0.09, 0.07, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def ar1_series(z, r):
    """The stationary AR(1) series with coefficient r and innovations z (in rows)."""
    z = np.array(z, dtype=float)
    z[0] = z[0] / np.sqrt(1 - r**2)
    return lfilter([1.], [1., -r], z, axis=0)

def sim_data(n):
    X = ar1_series(np.sqrt(1 - phi**2) * rng.standard_normal((n, p)) @ C_half, phi) * scale
    eps = ar1_series(sigma * rng.standard_normal(n), rho)
    return X, 1 + X @ beta + eps

X, y = sim_data(n)
```

The errors are independent of the features, so the target of the
selected model $E$, the population least squares regression of $y$ on
$X_E$, is exact: with $\Sigma_x$ the covariance of $x_t$,
$\beta_E = (\Sigma_x)_{EE}^{-1}(\Sigma_x)_{E\cdot}\,\beta$.

```{code-cell} ipython3
Sigma_x = C * np.outer(scale, scale)

def target(E):
    return np.linalg.solve(Sigma_x[np.ix_(E, E)], Sigma_x[E] @ beta)
```

## Estimating the AR(1) model

We estimate the AR(1) model from the residuals $r = My$ of least
squares on all $p$ features, $M = I - X_1(X_1^TX_1)^{-1}X_1^T$, by the
Yule–Walker equations, which match the lag-0 and lag-1 autocovariances.
Applied to the residuals as if they were the errors, they give the
residuals' lag-1 autocorrelation, a poor estimate of $\rho$ here: the features
are smooth series, and least squares on them absorbs the slowly varying
part of the errors, so the residuals are less autocorrelated than the
errors. For these data it averages about 0.68 for $n = 2000$ and 0.39
for $n = 200$, when $\rho = 0.7$.

Instead we use the autocovariances of the residuals themselves: $r = M\epsilon$
has covariance $M\Sigma_\epsilon M$, so the Yule–Walker equations for the
residuals are

$$\sum_t r_t^2 = \gamma_0\,\mathrm{tr}\big(M R(\rho)\big), \qquad \sum_{t>1} r_t r_{t-1} = \gamma_0\,\mathrm{tr}\big(L M R(\rho) M\big),$$

where $R(\rho)$ is the AR(1) correlation matrix, with entries
$\rho^{|t-s|}$, $L$ has $\tfrac12$ above and below its diagonal, and
$\gamma_0 = \sigma^2/(1-\rho^2)$ is the marginal variance. The ratio
of the two equations gives $\hat\rho$, and the first then gives
$\hat\gamma_0$. The traces need only products of $R(\rho)$, a Toeplitz
matrix, with $X_1$. Then $\hat\Sigma_\epsilon = \hat\gamma_0 R(\hat\rho)$,
and the covariance of the score is estimated by
$X_1^T\hat\Sigma_\epsilon X_1/n^2$.

```{code-cell} ipython3
def with_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def ar1_fit(X, y):
    """Yule-Walker estimates (rho_hat, sigma2_hat) from the residuals, corrected for the fit."""
    X1 = with_intercept(X)
    n = len(y)
    B = np.linalg.inv(X1.T @ X1)
    r = y - X1 @ (B @ (X1.T @ y))
    zero = np.zeros((1, X1.shape[1]))
    LX1 = 0.5 * (np.vstack([X1[1:], zero]) + np.vstack([zero, X1[:-1]]))
    XLX = X1.T @ LX1

    def expected(rho):
        # tr(M R) and tr(L M R M)
        c = rho ** np.arange(n)
        XRX = X1.T @ matmul_toeplitz((c, c), X1)
        XRLX = X1.T @ matmul_toeplitz((c, c), LX1)
        e0 = n - np.trace(B @ XRX)
        e1 = (n - 1) * rho - 2 * np.trace(B @ XRLX) + np.trace(B @ XRX @ B @ XLX)
        return e0, e1

    ratio = (r[1:] @ r[:-1]) / (r @ r)
    rho_hat = brentq(lambda rho: expected(rho)[1] / expected(rho)[0] - ratio, -0.99, 0.99)
    gamma0_hat = (r @ r) / expected(rho_hat)[0]
    return rho_hat, gamma0_hat * (1 - rho_hat**2)

def ar1_covariance(n, rho, sigma2):
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return sigma2 / (1 - rho**2) * rho ** lags

def score_covariance(X, Sigma_eps):
    X1 = with_intercept(X)
    return X1.T @ Sigma_eps @ X1 / X.shape[0]**2

rho_hat, sigma2_hat = ar1_fit(X, y)
rho_hat, sigma2_hat
```

For these data the score's variance is several times what independent
errors with the same marginal variance would give:

```{code-cell} ipython3
Sigma_eps = ar1_covariance(n, rho_hat, sigma2_hat)
iid = sigma2_hat / (1 - rho_hat**2) * np.eye(n)
np.round(np.diag(score_covariance(X, Sigma_eps)) / np.diag(score_covariance(X, iid)), 2)[:10]
```

## The randomized lasso

`lasso` fits the lasso at one value of $\lambda$, with an optional
linear term `LinearTerm(coef=G, intercept=G_0)` adding
$G_0\beta_0 + G^T\beta$ to the objective. `relaxed_fit` refits a model
$E$ without a penalty, keeping the linear term. The randomized fit uses
tight convergence tolerances, as the conditional inference assumes the
lasso was solved exactly.

```{code-cell} ipython3
pi = 0.9
lam = 1.5 * sigma / np.sqrt(1 - rho**2) * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def lasso(X, y, linear_term=None, control=None):
    return GLMNet(lambda_values=np.array([lam]),
                  linear_term=linear_term,
                  control=control or GLMNetControl()).fit(X, y)

def relaxed_fit(X, y, E, linear_term):
    others = sorted(set(range(X.shape[1])) - set(E))
    return GLMNet(lambda_values=np.array([0.]),
                  exclude=others,
                  linear_term=linear_term,
                  control=tight).fit(X, y).coefs_[0][E]
```

## A perturbation from the fitted AR(1) model

Draw two independent series $\epsilon^*_1, \epsilon^*_2$ from the fitted
AR(1) model and set

$$e = \sqrt{\frac{1-\pi}{\pi}}\;\frac{\epsilon^*_1 - \epsilon^*_2}{\sqrt 2}, \qquad \omega = \frac1n X_1^T e.$$

Given the fit, $e$ has mean 0 and covariance
$\frac{1-\pi}{\pi}\hat\Sigma_\epsilon$, the serial correlation included,
so $\omega$ has $(1-\pi)/\pi$ times the estimated covariance of the
score, $X_1^T\hat\Sigma_\epsilon X_1/n^2$. This is the relation that
makes selection with the linear term $-\omega$, intercept entry
included, imitate selection on a fraction $\pi$ of the data. For least
squares the score is linear in $y$, so this linear term is exactly the
effect of the response $y + e$; the response itself is not changed.

```{code-cell} ipython3
def ar1_draw(n, rho, sigma2):
    return ar1_series(np.sqrt(sigma2) * rng.standard_normal(n), rho)

def perturbation(X, rho, sigma2):
    n = X.shape[0]
    e = np.sqrt((1 - pi) / pi) * (ar1_draw(n, rho, sigma2) - ar1_draw(n, rho, sigma2)) / np.sqrt(2)
    return LinearTerm(coef=-X.T @ e / n, intercept=-e.mean())
```

## Inference with the AR(1) covariance

Both ways of doing inference carry over, with $\hat\Sigma_\epsilon$ in
the variances.

- **Combination.** $T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$,
  where $\hat\beta_E$ is the least squares fit of $y$ on $X_E$ and
  $\hat\beta^*_E$ its fit with the linear term $-\omega$. Its standard
  errors are the sandwich ones,
  $(X_{E1}^TX_{E1})^{-1}X_{E1}^T\hat\Sigma_\epsilon X_{E1}(X_{E1}^TX_{E1})^{-1}$
  with $X_{E1} = [\mathbf 1, X_E]$, divided by $\sqrt{1-\pi}$.
- **Carving.** The selection problem is the lasso problem for $y$ with
  the linear term $-\omega$. `glmstar_inference_problem` builds it from
  the randomized fit; its default covariance of the score,
  $\hat\sigma^2_\epsilon X_1^TX_1/n^2$, is for independent errors, so we
  pass `Sigma` $= X_1^T\hat\Sigma_\epsilon X_1/n^2$, with
  `scalar_noise = (1 - pi) / pi`.

The selected model $E$ is the one `lassoinf` conditions on, which
counts a coefficient that is numerically 0 as 0. Both methods use it.

`randomized_inference` does one randomized selection and returns both
sets of intervals. With `dependent=False` it ignores the dependence: it
draws $e$ as white noise with the estimated marginal variance
$\hat\sigma^2/(1-\hat\rho^2)$, and uses that diagonal covariance in the
inference.

```{code-cell} ipython3
def sandwich(X_E, Sigma_eps):
    X1 = with_intercept(X_E)
    B = np.linalg.inv(X1.T @ X1)
    return B @ X1.T @ Sigma_eps @ X1 @ B

def least_squares(X_E, y):
    return np.linalg.lstsq(with_intercept(X_E), y, rcond=None)[0]

def randomized_inference(X, y, rho_hat, sigma2_hat, dependent=True):
    n = X.shape[0]
    if dependent:
        omega = perturbation(X, rho_hat, sigma2_hat)
        Sigma_eps = ar1_covariance(n, rho_hat, sigma2_hat)
    else:
        omega = perturbation(X, 0., sigma2_hat / (1 - rho_hat**2))
        Sigma_eps = sigma2_hat / (1 - rho_hat**2) * np.eye(n)
    fit = lasso(X, y, linear_term=omega, control=tight)
    if not np.any(fit.coefs_[0]):
        return [], None

    # carving
    info = glmstar_inference_problem(fit, X, y, scalar_noise=(1 - pi) / pi,
                                     Sigma=score_covariance(X, Sigma_eps))
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_
    summary = summary.loc[summary.index > 0]       # row 0 is the intercept
    E = summary.index.values - 1
    if len(E) == 0:
        return E, None
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    beta_hat = least_squares(X[:, E], y)[1:]
    T = (beta_hat - pi * relaxed_fit(X, y, E, omega)) / (1 - pi)
    se = np.sqrt(np.diag(sandwich(X[:, E], Sigma_eps))[1:] / (1 - pi))
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

### One data set

```{code-cell} ipython3
E, intervals = randomized_inference(X, y, rho_hat, sigma2_hat)
table = pd.concat({m: pd.DataFrame({'estimate': est, 'lower': lower, 'upper': upper},
                                   index=[f'X{j}' for j in E])
                   for m, (est, lower, upper) in intervals.items()}, axis=1)
table['target'] = target(E)
table.round(3)
```

## What ignoring the dependence does

Suppose we draw $e$ as white noise and use independent errors in the
inference. Write $\Sigma$ for the covariance of the score,
$X_1^T\Sigma_\epsilon X_1/n^2$, and $\Sigma_0$ for its value under
independence. Then $\mathrm{Var}(\omega) = \frac{1-\pi}{\pi}\Sigma_0$
instead of $\frac{1-\pi}{\pi}\Sigma$.

- **Combination.** $T_E$ was independent of the selection because
  $S - \frac{\pi}{1-\pi}\omega$ is uncorrelated with the selection
  score $S + \omega$. Now their covariance is $\Sigma - \Sigma_0$, which
  is not 0, and the standard errors are too small.
- **Carving.** `lassoinf` takes the score's covariance to be $\Sigma_0$
  when it is $\Sigma$, so the intervals are too short, and the
  randomization is smaller relative to the score than `scalar_noise`
  says.

## Calibration

We repeat the experiment and compare six methods:

- **naive**: select with the lasso on the data, then use the least
  squares intervals of the selected model, with the AR(1) sandwich
  standard errors. This accounts for the dependence but not for the
  selection;
- **split**: select on the first 90% of the series, then form intervals
  on the last 10%, with the AR(1) model estimated there. A random split
  would leave the two parts dependent; contiguous blocks are nearly
  independent;
- **combination** and **carved**: as above, with $\hat\Sigma_\epsilon$;
- **combination (iid)** and **carved (iid)**: the same, ignoring the
  dependence.

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

    def sandwich_intervals(X, y, E):
        rho_hat, sigma2_hat = ar1_fit(X, y)
        b = least_squares(X[:, E], y)[1:]
        se = np.sqrt(np.diag(sandwich(X[:, E], ar1_covariance(X.shape[0], rho_hat, sigma2_hat)))[1:])
        return b - q * se, b + q * se

    n_sel = int(pi * n)
    for _ in range(n_rep):
        X, y = sim_data(n)
        rho_hat, sigma2_hat = ar1_fit(X, y)

        E = np.nonzero(lasso(X, y).coefs_[0])[0]
        if len(E):
            record('naive', E, *sandwich_intervals(X, y, E))

        E = np.nonzero(lasso(X[:n_sel], y[:n_sel]).coefs_[0])[0]
        if len(E):
            record('split', E, *sandwich_intervals(X[n_sel:], y[n_sel:], E))

        for suffix, dependent in [('', True), (' (iid)', False)]:
            E, intervals = randomized_inference(X, y, rho_hat, sigma2_hat, dependent=dependent)
            if len(E):
                for method, (_, lower, upper) in intervals.items():
                    record(method + suffix, E, lower, upper)

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py autocorrelated_errors`
and saved in `docs/results/autocorrelated_errors.csv`.

```{code-cell} ipython3
results = load_results('autocorrelated_errors')
display(provenance(results))
display(reproduce(results))

methods = ['naive', 'split', 'combination', 'carved', 'combination (iid)', 'carved (iid)']
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

The naive intervals fail for the selected nulls and undercover for the
weak signals, because the variables were chosen with the same data used
for inference. With $\hat\Sigma_\epsilon$, the combination and carved
intervals are calibrated for each kind of signal, and the carved ones
are much shorter. As in the other examples, for the strong signals they
are close to the naive intervals, and for the weak signals and the
nulls they widen but stay shorter than the combination intervals.

The split intervals undercover slightly. They have to estimate the
AR(1) model from the held-out 10% of the series alone, 200 observations
of strongly autocorrelated features, and the long-run variance of the
score is sensitive to the estimate of $\rho$. The randomized methods
estimate it from the whole series, and hold no data out.

Ignoring the dependence breaks the randomized methods. The carved
intervals are about half as long as with $\hat\Sigma_\epsilon$, and
undercover for every kind of signal. The combination intervals undercover less for the strong
signals, whose selection is nearly certain; for the nulls, the leftover
correlation $\Sigma - \Sigma_0$ with the selection matters, and they
undercover badly.

The calibration relies on the AR(1) model for the errors: $e$ has the
covariance of the errors under the fitted model, and the inference uses
the same covariance for the score. If the dependence has another form,
a block bootstrap of the residuals can supply the randomization instead
({ref}`Fry, Panigrahi and Taylor, 2025 <ar1-refs>`).

(ar1-refs)=
## References

- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
