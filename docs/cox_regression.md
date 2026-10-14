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

# Resampling the Cox score

This vignette compares ways of resampling the score of Cox's
proportional hazards model ({ref}`Cox, 1972 <cox-refs>`), with hazard
$\lambda(t \mid x) = \lambda_0(t)\exp(x^T\beta)$, when many event times
are tied and the ties are handled by Efron's approximation
({ref}`Efron, 1977 <cox-refs>`). It then uses each one to randomize the
lasso's selection, and does inference after selection with `lassoinf`,
as in [Score bootstraps with a linear term](score_bootstrap.md).

Resampling the score instead of the data goes back to Parzen, Wei and
Ying ({ref}`1994 <cox-refs>`), and is the idea of the score bootstrap
({ref}`Kline and Santos, 2012 <cox-refs>`). The other vignettes
randomize the selection with a perturbation $\omega$
of the score whose covariance is $(1-\pi)/\pi$ times that of the score,
so that selecting with it imitates selecting on a random fraction $\pi$
of the data. For Cox's model there are several natural resampling
schemes, and they do not all have the variance of the score. The
calibration at the end shows what that does to the inference.

```{code-cell} ipython3
import warnings

import numpy as np
import pandas as pd
from coxdev import CoxDeviance
from scipy.stats import norm

from glmnet import LinearTerm
from glmnet.cox import CoxFamily, CoxNetIRLS
from glmnet.glmnet import GLMNetControl
from glmnet.inference import GLMProblem
from glmnet.inference.glm_inference import GLMInferenceProblem
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
```

## Data

We simulate $n = 3000$ subjects with exponential event times, hazard
$\exp(x^T\beta)$, and uniform censoring on $[0, 3]$. As in the other
examples, there are two strong signals, six weak ones and 22 nulls. The
features are correlated, with standard deviations from 0.2 to 5, and the
effects are set per standard deviation of each feature.

The observed times are then rounded up to a grid of 25 times, the
quantiles of the observed times in a large sample. This is how ties
arise when time is recorded in days or months: each of the 25 times has
many events.

```{code-cell} ipython3
rng = np.random.default_rng(5)
n, p, rho = 3000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.4, -0.3, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def continuous_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    T = rng.exponential(1 / np.exp(X @ beta))
    C = rng.uniform(0, 3, n)
    return X, np.minimum(T, C), (T <= C).astype(float)

# the grid: quantiles of the observed times, ending at the largest censoring time
grid = np.quantile(continuous_data(100000)[1], np.linspace(0, 1, 26)[1:])
grid[-1] = 3.

def sim_data(n):
    X, time, status = continuous_data(n)
    return X, grid[np.searchsorted(grid, time)], status

X, time, status = sim_data(n)
event_times, ties = np.unique(time[status == 1], return_counts=True)
print(f'{int(status.sum())} events at {len(event_times)} times, '
      f'from {ties.min()} to {ties.max()} at each')
```

## The partial likelihood and its score

`CoxNetIRLS` fits the lasso to the Efron partial likelihood. Its loss is
the negative log partial likelihood divided by $n$. `coxdev`, which
`CoxNetIRLS` uses, computes its gradient and Hessian with respect to the
linear predictor $\eta = X\beta$, with the weights $1/n$. Newton's
method on them fits the unpenalized Cox regression, and the inverse
Hessian divided by $n$ estimates the covariance of the fit.

```{code-cell} ipython3
def loss_gradient_hessian(X, time, status, b):
    """Gradient and Hessian of the Efron loss at coefficients b."""
    n = X.shape[0]
    cox = CoxDeviance(event=time, status=status, tie_breaking='efron')
    eta = X @ b
    gradient = X.T @ cox(eta, np.ones(n) / n).gradient / 2
    info = cox.information(eta, np.ones(n) / n)
    hessian = X.T @ np.column_stack([info @ X[:, j] for j in range(X.shape[1])])
    return gradient, hessian

def cox_fit(X, time, status, tol=1e-10):
    """The unpenalized Cox regression with Efron ties: coefficients and standard errors."""
    b = np.zeros(X.shape[1])
    for _ in range(50):
        gradient, hessian = loss_gradient_hessian(X, time, status, b)
        step = np.linalg.solve(hessian, gradient)
        b = b - step
        if np.abs(step).max() < tol:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(hessian)) / X.shape[0])
```

## Resampling the score

At the $k$-th event time $t_k$, with risk set $\mathcal R_k$, $d_k$
tied failures and fitted risks $\exp(\hat\eta_i)$, write
$p_{ki} \propto \exp(\hat\eta_i)$, $i \in \mathcal R_k$, for the
probability that subject $i$ is the next to fail. The total score is
$U = \sum_k \big(\sum_{i \text{ fails at } t_k} x_i - d_k \bar x_k\big)$,
roughly, with $\bar x_k = \sum_i p_{ki} x_i$, and its covariance is
about the information $nQ$, with $Q$ the Hessian of the loss. Each
scheme below draws a $U^*$ with mean 0 from the fitted model, and the
perturbation is $\omega = \sqrt{(1-\pi)/\pi}\;U^*/n$.

- **Multinomial (one draw).** Draw the $d_k$ failures at each $t_k$
  with replacement, $N_k \sim \mathrm{Multinomial}(d_k, p_k)$, and
  center: $U^* = \sum_k \sum_i x_i(N_{ki} - d_k p_{ki})$. This is the
  "weird bootstrap" of Andersen, Borgan, Gill and Keiding
  ({ref}`1993 <cox-refs>`), compared with other bootstraps for Cox's
  model by Burr ({ref}`1994 <cox-refs>`). Its covariance is the
  information of Breslow's approximation for ties
  ({ref}`Breslow, 1974 <cox-refs>`).
- **Multinomial.** The difference of two independent multinomial
  draws, divided by $\sqrt 2$, as in the other vignettes: the same
  covariance, no centering, and a symmetric distribution.
- **Bernoulli.** Andersen, Borgan, Gill and Keiding's original form
  ({ref}`1993 <cox-refs>`):
  each subject at risk fails independently with probability
  $\pi_{ki} = \min(1, d_k p_{ki})$, centered by $\bar x_k$; again the
  difference of two draws over $\sqrt 2$. The variance of each jump is
  $\pi_{ki}(1-\pi_{ki})$ rather than $\pi_{ki}$.
- **Wallenius.** Draw the $d_k$ failures without replacement, one at a
  time, each with probabilities $\propto \exp(\hat\eta_i)$ among the
  subjects not yet drawn: Wallenius' noncentral hypergeometric
  distribution ({ref}`Wallenius, 1963 <cox-refs>`). The $d_k$ largest
  values of $\hat\eta_i$ plus independent Gumbel noise are such a draw
  ({ref}`Efraimidis and Spirakis, 2006 <cox-refs>`;
  {ref}`Kool, van Hoof and Welling, 2019 <cox-refs>`). Each subject fails
  at most once, as in the data. Again two draws.
- **Wild.** Multiply each subject's score residual $\hat U_i$ (its
  contribution to the score, centered within the risk sets, with
  Efron's weights) by an independent standard normal $G_i$:
  $U^* = \sum_i G_i \hat U_i$. This is a wild bootstrap
  ({ref}`Wu, 1986 <cox-refs>`), as Lin, Wei and Ying
  ({ref}`1993 <cox-refs>`) and Lin, Fleming and Wei
  ({ref}`1994 <cox-refs>`) used for Cox's model; see Beyersmann, Di
  Termini and Pauly ({ref}`2013 <cox-refs>`) for its theory in survival
  analysis. Its covariance is the sandwich
  $\sum_i \hat U_i \hat U_i^T$ ({ref}`Lin and Wei, 1989 <cox-refs>`),
  which does not rely on the model being right.

```{code-cell} ipython3
SCHEMES = ['multinomial (one draw)', 'multinomial', 'bernoulli', 'wallenius', 'wild']

def risk_sets(time, status):
    """For each event time: the risk set, the subjects who fail, and their number."""
    times, d = np.unique(time[status == 1], return_counts=True)
    return [(np.flatnonzero(time >= t_k), np.flatnonzero((time == t_k) & (status == 1)), d_k)
            for t_k, d_k in zip(times, d)]

def score_residuals(X, eta, sets):
    """Efron score residuals: sum_k sum_r (dN_ik / d_k - p_ikr) (x_i - xbar_kr)."""
    U = np.zeros_like(X)
    w = np.exp(eta - eta.max())
    for idx, fail, d_k in sets:
        tied = np.isin(idx, fail)
        X_k = X[idx]
        for r in range(d_k):
            # Efron: the r-th of the d_k tied failures sees the tied subjects with weight 1 - r / d_k
            w_r = w[idx] * (1 - (r / d_k) * tied)
            p_r = w_r / w_r.sum()
            U[idx] += (tied / d_k - p_r)[:, None] * (X_k - p_r @ X_k)
    return U

def score_draw(X, eta, sets, scheme, residuals=None):
    """One draw U* of the resampled total score."""
    if scheme == 'wild':
        return rng.standard_normal(X.shape[0]) @ residuals
    U = np.zeros(X.shape[1])
    for idx, fail, d_k in sets:
        X_k = X[idx]
        p_k = np.exp(eta[idx] - eta[idx].max())
        p_k /= p_k.sum()
        if scheme == 'multinomial (one draw)':
            U += (rng.multinomial(d_k, p_k) - d_k * p_k) @ X_k
        elif scheme == 'multinomial':
            U += (rng.multinomial(d_k, p_k) - rng.multinomial(d_k, p_k)) @ X_k / np.sqrt(2)
        elif scheme == 'bernoulli':
            pi_k = np.minimum(1, d_k * p_k)
            U += (rng.binomial(1, pi_k) - rng.binomial(1, pi_k)) @ (X_k - p_k @ X_k) / np.sqrt(2)
        elif scheme == 'wallenius':
            for sign in (1, -1):
                top = np.argpartition(-(eta[idx] + rng.gumbel(size=len(idx))), d_k - 1)[:d_k]
                U += sign * X_k[top].sum(0) / np.sqrt(2)
    return U
```

### Which has the variance of the score?

On this data set, at the fitted $\hat\beta$, we compare the variance of
each scheme's draws with the Efron information, coordinate by
coordinate:

```{code-cell} ipython3
b_hat = cox_fit(X, time, status)[0]
eta_hat, sets = X @ b_hat, risk_sets(time, status)
residuals = score_residuals(X, eta_hat, sets)
information = n * loss_gradient_hessian(X, time, status, b_hat)[1]

ratios = {scheme: np.diag(np.cov([score_draw(X, eta_hat, sets, scheme, residuals)
                                  for _ in range(1000)], rowvar=False)) / np.diag(information)
          for scheme in SCHEMES}
pd.DataFrame(ratios).describe().loc[['mean', 'min', 'max']].round(3)
```

The Efron information is the right target: over many data sets, at the
true $\beta$, the score itself varies as the Efron information says.

```{code-cell} ipython3
scores, infos = [], []
for _ in range(200):
    X_s, time_s, status_s = sim_data(n)
    gradient, hessian = loss_gradient_hessian(X_s, time_s, status_s, beta)
    scores.append(-n * gradient)
    infos.append(n * hessian)
ratio = np.diag(np.cov(scores, rowvar=False)) / np.diag(np.mean(infos, axis=0))
print(f'variance of the score / Efron information: mean {ratio.mean():.3f}, '
      f'from {ratio.min():.3f} to {ratio.max():.3f} (each with Monte Carlo error about 0.1)')
```

The multinomial draws, with one draw or two, and the wild draws have
the variance of the score. The Bernoulli and Wallenius draws have about
10% less. Both draw without replacement, in effect: in the Bernoulli
scheme each subject fails at most once, and its jumps have variance
$\pi_{ki}(1-\pi_{ki})$. When $d_k$ is a sizable fraction of the risk set,
as here, this removes variance much as a finite-population correction
does. Efron's approximation, and the score of the tied data, have
almost none of that reduction.

## The randomized lasso and the inference

`cox_lasso` fits the lasso at one value of $\lambda$, possibly with a
linear term. Cox's model has no intercept, so the linear term has an
entry for each feature only. With `relax=True` it also refits the
selected model without a penalty, keeping the linear term. As in the
other vignettes, $\lambda$ is set for a lasso on $\pi n$ observations.

`glmnet.inference` builds the selection problem for GLM families. For
Cox's model we build it from the gradient and Hessian above, and
`lassoinf` does the rest:

- **The selection problem** is the lasso problem for the data with the
  linear term $-\omega$. At the randomized fit $\hat\beta$, its gradient
  is the gradient of the loss minus $\omega$, and its Hessian $Q$ is at
  the relaxed fit: one Newton step for the loss (and its linear term)
  on the selected coordinates.
- **The score of the full data** linearizes the unperturbed gradient at
  one Newton step from $\hat\beta$ on the selected coordinates,
  $\tilde\beta$: $Z = -\nabla(\tilde\beta) + Q\tilde\beta$, with
  covariance $Q/n$. The randomization has `scalar_noise`
  $= (1-\pi)/\pi$.

The combination is as before:
$T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$, with $\hat\beta_E$
the Cox regression on $X_E$ and $\hat\beta^*_E$ the relaxed fit of the
randomized lasso, and standard errors those of $\hat\beta_E$ divided by
$\sqrt{1-\pi}$.

```{code-cell} ipython3
q = norm.ppf(0.95)  # for nominal 90% intervals
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def lam(pi):
    return 0.7 * np.sqrt(2 * np.log(p) / (pi * n))

def cox_lasso(X, time, status, pi, linear_term=None, relax=False, control=None):
    return CoxNetIRLS(family=CoxFamily(tie_breaking='efron'),
                      lambda_values=np.array([lam(pi)]),
                      linear_term=linear_term,
                      relax=relax,
                      control=control or GLMNetControl()).fit(
                          X, pd.DataFrame({'event': time, 'status': status}))

def newton_step(X, time, status, b, linear_term=0.):
    """One Newton step for the loss (plus a linear term) on the nonzero coordinates of b."""
    active = np.nonzero(b)[0]
    gradient, hessian = loss_gradient_hessian(X, time, status, b)
    b = b.copy()
    b[active] -= np.linalg.solve(hessian[np.ix_(active, active)],
                                 (gradient + linear_term)[active])
    return b

def randomized_inference(X, time, status, pi, scheme='multinomial'):
    n = X.shape[0]
    eta = X @ cox_fit(X, time, status)[0]          # the pilot fit
    sets = risk_sets(time, status)
    residuals = score_residuals(X, eta, sets) if scheme == 'wild' else None
    omega = np.sqrt((1 - pi) / pi) * score_draw(X, eta, sets, scheme, residuals) / n
    fit = cox_lasso(X, time, status, pi, linear_term=LinearTerm(coef=-omega),
                    relax=True, control=tight)
    b = fit.coefs_[0]
    if not np.any(b):
        return [], None

    # carving
    gradient = loss_gradient_hessian(X, time, status, b)[0]
    Q = loss_gradient_hessian(X, time, status, newton_step(X, time, status, b, -omega))[1]
    problem = GLMProblem(beta_hat=b, G_hat=gradient - omega, Q_hat=Q,
                         D=lam(pi) * np.asarray(fit.design_.scaling_),
                         L=np.full(p, -np.inf), U=np.full(p, np.inf),
                         fit_intercept=False, linear_term=-omega)
    b_tilde = newton_step(X, time, status, b)
    Z_full = -loss_gradient_hessian(X, time, status, b_tilde)[0] + Q @ b_tilde
    info = GLMInferenceProblem(problem=problem, Z_full=Z_full, Sigma=Q / n,
                               scalar_noise=(1 - pi) / pi, dispersion=1.)
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_
    E = summary.index.values
    if len(E) == 0:
        return E, None
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    b_E, se_E = cox_fit(X[:, E], time, status)
    T = (b_E - pi * fit.relaxed_coefs_[0][E]) / (1 - pi)
    se = se_E / np.sqrt(1 - pi)
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the Cox regression of the
population on $X_E$, with the same grid of times and Efron ties. We
approximate them by fitting each selected model to one large sample of a
million subjects, so that their own error is small, about a twentieth of
the standard error of a fit to $n = 3000$:

```{code-cell} ipython3
pop_X, pop_time, pop_status = sim_data(1000000)
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        targets[key] = cox_fit(pop_X[:, E], pop_time, pop_status)[0]
    return targets[key]
```

### One data set

With the multinomial draws and $\pi = 0.9$:

```{code-cell} ipython3
E, intervals = randomized_inference(X, time, status, 0.9)
table = pd.concat({m: pd.DataFrame({'estimate': est, 'lower': lower, 'upper': upper},
                                   index=[f'X{j}' for j in E])
                   for m, (est, lower, upper) in intervals.items()}, axis=1)
table['target'] = target(E)
table.round(3)
```

## What a perturbation with too little variance does

Write $\Sigma$ for the covariance of the score and $\Sigma_*$ for that of
a scheme's draws, so that
$\mathrm{Var}(\omega) = \frac{1-\pi}{\pi}\Sigma_*$.

- **Combination.** $S - \frac{\pi}{1-\pi}\omega$ is uncorrelated with
  the selection score $S + \omega$ only when $\Sigma_* = \Sigma$; for
  the Bernoulli and Wallenius draws their covariance is
  $\Sigma - \Sigma_* \approx 0.1\,\Sigma$.
- **Carving.** `lassoinf` takes the randomization to have
  $(1-\pi)/\pi$ times the variance of the score, and credits the
  selection with more randomization than it had.

Both effects grow with the randomization. At $\pi = 0.9$ the
perturbation has a ninth of the variance of the score, and a 10%
shortfall in it is a 1% shortfall in the selection score's; at
$\pi = 0.5$ it has as much as the score.

## Calibration

We repeat the experiment at $\pi = 0.9$ and $\pi = 0.5$ and compare:

- **naive**: select with the lasso on the data, then use the Cox
  regression intervals of the selected model on the same data;
- **split**: select on a random fraction $\pi$ of the subjects, then
  form intervals on the others;
- **combination** and **carved**, with each of the multinomial (two
  draws), Bernoulli, Wallenius and wild perturbations.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length per standard
deviation of the feature.

```{code-cell} ipython3
def calibration(n_rep=300):
    rows = []

    def record(pi, method, scheme, E, lower, upper):
        for j, lo, hi, t in zip(E, lower, upper, target(E)):
            rows.append({'pi': pi, 'method': method, 'scheme': scheme, 'signal': signal[j],
                         'covered': lo <= t <= hi, 'length': (hi - lo) * scale[j]})

    def cox_intervals(X, time, status, E):
        b, se = cox_fit(X[:, E], time, status)
        return b - q * se, b + q * se

    for _ in range(n_rep):
        X, time, status = sim_data(n)
        for pi in [0.9, 0.5]:
            E = np.nonzero(cox_lasso(X, time, status, pi).coefs_[0])[0]
            if len(E):
                record(pi, 'naive', '', E, *cox_intervals(X, time, status, E))

            idx = rng.permutation(n)
            sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
            E = np.nonzero(cox_lasso(X[sel], time[sel], status[sel], pi).coefs_[0])[0]
            if len(E):
                record(pi, 'split', '', E,
                       *cox_intervals(X[held_out], time[held_out], status[held_out], E))

            for scheme in ['multinomial', 'bernoulli', 'wallenius', 'wild']:
                E, intervals = randomized_inference(X, time, status, pi, scheme)
                if len(E):
                    for method, (_, lower, upper) in intervals.items():
                        record(pi, method, scheme, E, lower, upper)

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py cox_regression`
and saved in `docs/results/cox_regression.csv`.

```{code-cell} ipython3
results = load_results('cox_regression')
display(provenance(results))
display(reproduce(results))

results['method'] = pd.Categorical(results['method'], ['naive', 'split', 'combination', 'carved'])
results['scheme'] = pd.Categorical(results['scheme'],
                                   ['', 'multinomial', 'bernoulli', 'wallenius', 'wild'])
results['signal'] = pd.Categorical(results['signal'], ['strong', 'weak', 'null'])
summary = (results.groupby(['pi', 'method', 'scheme'], observed=True)
                  .agg(intervals=('covered', 'size'),
                       coverage=('covered', 'mean'),
                       mean_length=('length', 'mean')))
summary.round(3)
```

By the strength of the signal, the coverage:

```{code-cell} ipython3
(results.groupby(['pi', 'method', 'scheme', 'signal'], observed=True)['covered'].mean()
        .unstack('signal').round(3))
```

DISCUSSION

(cox-refs)=
## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the
  Royal Statistical Society, Series B*, 34(2), 187–220.
- Efron, B. (1977). The efficiency of Cox's likelihood function for
  censored data. *Journal of the American Statistical Association*,
  72(359), 557–565.
- Breslow, N. E. (1974). Covariance analysis of censored survival data.
  *Biometrics*, 30(1), 89–99.
- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993).
  *Statistical Models Based on Counting Processes*. Springer.
- Burr, D. (1994). A comparison of certain bootstrap confidence
  intervals in the Cox model. *Journal of the American Statistical
  Association*, 89(428), 1290–1302.
- Wallenius, K. T. (1963). *Biased sampling: the noncentral
  hypergeometric probability distribution*. PhD thesis, Stanford
  University.
- Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling
  with a reservoir. *Information Processing Letters*, 97(5), 181–185.
- Kool, W., van Hoof, H., & Welling, M. (2019). Stochastic beams and
  where to find them: the Gumbel-top-k trick for sampling sequences
  without replacement. *Proceedings of the 36th International
  Conference on Machine Learning*, PMLR 97, 3499–3508.
- Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods
  in regression analysis. *The Annals of Statistics*, 14(4), 1261–1295.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox
  proportional hazards model. *Journal of the American Statistical
  Association*, 84(408), 1074–1078.
- Lin, D. Y., Wei, L. J., & Ying, Z. (1993). Checking the Cox model with
  cumulative sums of martingale-based residuals. *Biometrika*, 80(3),
  557–572.
- Lin, D. Y., Fleming, T. R., & Wei, L. J. (1994). Confidence bands for
  survival curves under the proportional hazards model. *Biometrika*,
  81(1), 73–81.
- Beyersmann, J., Di Termini, S., & Pauly, M. (2013). Weak convergence
  of the wild bootstrap for the Aalen–Johansen estimator of the
  cumulative incidence function of a competing risk. *Scandinavian
  Journal of Statistics*, 40(3), 387–402.
- Parzen, M. I., Wei, L. J., & Ying, Z. (1994). A resampling method
  based on pivotal estimating functions. *Biometrika*, 81(2), 341–350.
  <https://doi.org/10.1093/biomet/81.2.341>
- Kline, P., & Santos, A. (2012). A score based approach to wild
  bootstrap inference. *Journal of Econometric Methods*, 1(1), 23–41.
  <https://doi.org/10.1515/2156-6674.1006>
- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
