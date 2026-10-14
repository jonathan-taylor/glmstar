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

# A misspecified Cox model

[Resampling the Cox score](cox_regression.md) fitted Cox's model to data
from Cox's model. Here the log hazard has a term the model leaves out.
The fitted coefficients then estimate a projection: the coefficients of
the Cox model closest to the truth, in the sense of the partial
likelihood ({ref}`Struthers and Kalbfleisch, 1986 <coxmis-refs>`). Inference for the projection is still possible, but the
score no longer varies as the information says. Its variance is the
sandwich ({ref}`Lin and Wei, 1989 <coxmis-refs>`).

This matters for the randomized selection too. The multinomial draws of
the score come from the fitted model, so their covariance is the
information. The wild draws multiply the subjects' score residuals by
independent normals, so their covariance is the sandwich. We compare the
inference with each, and with the model-based or the sandwich covariance
in `lassoinf`.

```{code-cell} ipython3
import warnings

import numpy as np
import pandas as pd
from coxdev import CoxDeviance
from scipy.stats import norm

from glmnet import LinearTerm
from glmnet.cox import CoxFamily, CoxFamilySpec, CoxNetIRLS
from glmnet.glmnet import GLMNetControl
from glmnet.inference import GLMProblem
from glmnet.inference.glm_inference import GLMInferenceProblem
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
```

## Data

The design is that of [Resampling the Cox score](cox_regression.md):
$n = 3000$ subjects, two strong signals, six weak ones and 22 nulls,
correlated features with standard deviations from 0.2 to 5, uniform
censoring on $[0, 3]$, and times rounded up to a grid of 25 times, so
that many events are tied. But the log hazard is

$$x^T\beta + \tfrac12\big(u_1^2 - 1\big) + \tfrac12\big(u_2^2 - 1\big),$$

with $u_1, u_2$ the two strong features in standard units. The Cox model
we fit is linear in $x$, so it misses the curvature in the two strong
features.

```{code-cell} ipython3
rng = np.random.default_rng(7)
n, p, rho = 3000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.4, -0.3, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def continuous_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    u = X[:, :2] / scale[:2]
    log_hazard = X @ beta + 0.5 * (u**2 - 1).sum(1)     # not linear in x
    T = rng.exponential(1 / np.exp(log_hazard))
    C = rng.uniform(0, 3, n)
    return X, np.minimum(T, C), (T <= C).astype(float)

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

## The fit, its score and the sandwich

As in [Resampling the Cox score](cox_regression.md), the loss is the
Efron negative log partial likelihood divided by $n$, with gradient and
Hessian from `coxdev`. The score residual $\hat U_i$ of subject $i$ is
its contribution to the score, centered within each risk set with
Efron's weights. The model-based covariance of the score is $Q/n$, with
$Q$ the Hessian; the sandwich is $\sum_i \hat U_i\hat U_i^T/n^2$.

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

def score_residuals(X, time, status, b):
    """Efron score residuals: sum_k sum_r (dN_ik / d_k - p_ikr) (x_i - xbar_kr)."""
    eta = X @ b
    w = np.exp(eta - eta.max())
    U = np.zeros_like(X)
    times, d = np.unique(time[status == 1], return_counts=True)
    for t_k, d_k in zip(times, d):
        idx = np.flatnonzero(time >= t_k)
        tied = (time[idx] == t_k) & (status[idx] == 1)
        X_k = X[idx]
        for r in range(d_k):
            w_r = w[idx] * (1 - (r / d_k) * tied)
            p_r = w_r / w_r.sum()
            U[idx] += (tied / d_k - p_r)[:, None] * (X_k - p_r @ X_k)
    return U

def cox_fit(X, time, status, sandwich=True, tol=1e-10):
    """
    The unpenalized Cox regression with Efron ties: coefficients, and model-based and
    (if sandwich) sandwich standard errors.
    """
    n = X.shape[0]
    b = np.zeros(X.shape[1])
    for _ in range(50):
        gradient, hessian = loss_gradient_hessian(X, time, status, b)
        step = np.linalg.solve(hessian, gradient)
        b = b - step
        if np.abs(step).max() < tol:
            break
    H_inv = np.linalg.inv(hessian)
    if not sandwich:
        return b, np.sqrt(np.diag(H_inv) / n), None
    U = score_residuals(X, time, status, b)
    return b, np.sqrt(np.diag(H_inv) / n), np.sqrt(np.diag(H_inv @ (U.T @ U / n**2) @ H_inv))
```

On this data set the sandwich standard errors of the two strong
coefficients are larger than the model-based ones:

```{code-cell} ipython3
b_hat, se_model, se_sandwich = cox_fit(X, time, status)
pd.DataFrame({'model': se_model, 'sandwich': se_sandwich, 'ratio': se_sandwich / se_model},
             index=[f'X{j}' for j in range(p)]).head(8).round(4)
```

The sandwich is right. Over many data sets the score at the projection,
fitted here to a large sample, varies by about the sandwich, not the
information:

```{code-cell} ipython3
pop_X, pop_time, pop_status = sim_data(1000000)
projection = cox_fit(pop_X, pop_time, pop_status, sandwich=False)[0]

scores, infos = [], []
for _ in range(200):
    X_s, time_s, status_s = sim_data(n)
    gradient, hessian = loss_gradient_hessian(X_s, time_s, status_s, projection)
    scores.append(-n * gradient)
    infos.append(n * hessian)
variance = np.diag(np.cov(scores, rowvar=False))
U = score_residuals(X, time, status, b_hat)
pd.DataFrame({'Var(score) / information': variance / np.diag(np.mean(infos, axis=0)),
              'sandwich / information (this data set)': np.diag(U.T @ U) / (n * np.diag(
                  loss_gradient_hessian(X, time, status, b_hat)[1]))},
             index=[f'X{j}' for j in range(p)]).head(8).round(2)
```

## Two perturbations

From the pilot fit $\hat\beta$ on all $p$ features, the perturbation is
$\omega = \sqrt{(1-\pi)/\pi}\;U^*/n$ with one of two kinds of draw,
both from the samplers of `coxdev` through `CoxFamilySpec.sample_score`:

- **multinomial** draws: at each event time the $d_k$ failures drawn
  with replacement from the risk set with probabilities
  $\propto\exp(x_i^T\hat\beta)$, centered at their expectation
  ({ref}`Andersen, Borgan, Gill and Keiding, 1993 <coxmis-refs>`).
  Their covariance is the (Breslow) information: it comes from the model.
- **wild** draws: $U^* = \sum_i G_i\hat U_i$ with independent standard
  normal $G_i$ ({ref}`Lin, Wei and Ying, 1993 <coxmis-refs>`). Their
  covariance is the sandwich: it comes from the residuals.

```{code-cell} ipython3
def score_draw(X, time, status, b, scheme):
    """One draw U* of the resampled total score, from the pilot fit b."""
    family = CoxFamilySpec(pd.DataFrame({'event': time, 'status': status}),
                           event_id='event', status_id='status', tie_breaking='efron')
    return X.T @ family.sample_score(X @ b, method=scheme, rng=rng)
```

## The inference

The randomized lasso, the selection problem and the full-data score are
as in [Resampling the Cox score](cox_regression.md). What changes is the
covariance of the score passed to `lassoinf`, and the standard errors of
the combination:

- **model-based**: multinomial draws, `Sigma` $= Q/n$, and model-based
  standard errors;
- **robust**: wild draws, `Sigma` the sandwich
  $\sum_i \hat U_i\hat U_i^T/n^2$ with the residuals at the full-data
  one-step fit $\tilde\beta$, and sandwich standard errors.

In each, the randomization has `scalar_noise` $= (1-\pi)/\pi$ relative to
`Sigma`, which holds when the draws and `Sigma` agree: the information
for the multinomial draws, the sandwich for the wild ones.

```{code-cell} ipython3
pi = 0.9
lam = 0.7 * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def cox_lasso(X, time, status, linear_term=None, relax=False, control=None):
    return CoxNetIRLS(family=CoxFamily(tie_breaking='efron'),
                      lambda_values=np.array([lam]),
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

def randomized_inference(X, time, status, robust):
    n = X.shape[0]
    pilot = cox_fit(X, time, status, sandwich=False)[0]
    omega = (np.sqrt((1 - pi) / pi)
             * score_draw(X, time, status, pilot, 'wild' if robust else 'multinomial') / n)
    fit = cox_lasso(X, time, status, linear_term=LinearTerm(coef=-omega), relax=True,
                    control=tight)
    b = fit.coefs_[0]
    if not np.any(b):
        return [], None

    # carving
    gradient = loss_gradient_hessian(X, time, status, b)[0]
    Q = loss_gradient_hessian(X, time, status, newton_step(X, time, status, b, -omega))[1]
    problem = GLMProblem(beta_hat=b, G_hat=gradient - omega, Q_hat=Q,
                         D=lam * np.asarray(fit.design_.scaling_),
                         L=np.full(p, -np.inf), U=np.full(p, np.inf),
                         fit_intercept=False, linear_term=-omega)
    b_tilde = newton_step(X, time, status, b)
    Z_full = -loss_gradient_hessian(X, time, status, b_tilde)[0] + Q @ b_tilde
    if robust:
        U = score_residuals(X, time, status, b_tilde)
        Sigma = U.T @ U / n**2
    else:
        Sigma = Q / n
    info = GLMInferenceProblem(problem=problem, Z_full=Z_full, Sigma=Sigma,
                               scalar_noise=(1 - pi) / pi, dispersion=np.nan)
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_
    E = summary.index.values
    if len(E) == 0:
        return E, None
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    b_E, se_model, se_sandwich = cox_fit(X[:, E], time, status)
    T = (b_E - pi * fit.relaxed_coefs_[0][E]) / (1 - pi)
    se = (se_sandwich if robust else se_model) / np.sqrt(1 - pi)
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the Cox regression of the
population on $X_E$, which we approximate by fitting each selected model
to the sample of a million subjects drawn above.

```{code-cell} ipython3
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        targets[key] = cox_fit(pop_X[:, E], pop_time, pop_status, sandwich=False)[0]
    return targets[key]
```

### One data set

```{code-cell} ipython3
tables = {}
for robust in [False, True]:
    E, intervals = randomized_inference(X, time, status, robust)
    tables['robust' if robust else 'model-based'] = pd.DataFrame(
        {'lower': intervals['carved'][1], 'upper': intervals['carved'][2],
         'target': target(E)}, index=[f'X{j}' for j in E])
pd.concat(tables, axis=1).round(3)
```

## Calibration

We repeat the experiment and compare, each with model-based and with
robust variances:

- **naive**: select with the lasso on the data, then use the Cox
  regression intervals of the selected model on the same data;
- **split**: select on a random 90% of the subjects, then form intervals
  on the other 10%;
- **combination** and **carved**: the multinomial draws with the
  information, or the wild draws with the sandwich.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length per standard
deviation of the feature.

```{code-cell} ipython3
def calibration(n_rep=300):
    rows = []

    def record(method, variance, E, lower, upper):
        for j, lo, hi, t in zip(E, lower, upper, target(E)):
            rows.append({'method': method, 'variance': variance, 'signal': signal[j],
                         'covered': lo <= t <= hi, 'length': (hi - lo) * scale[j]})

    def cox_intervals(method, X, time, status, E):
        b, se_model, se_sandwich = cox_fit(X[:, E], time, status)
        record(method, 'model-based', E, b - q * se_model, b + q * se_model)
        record(method, 'robust', E, b - q * se_sandwich, b + q * se_sandwich)

    for _ in range(n_rep):
        X, time, status = sim_data(n)

        E = np.nonzero(cox_lasso(X, time, status).coefs_[0])[0]
        if len(E):
            cox_intervals('naive', X, time, status, E)

        idx = rng.permutation(n)
        sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
        E = np.nonzero(cox_lasso(X[sel], time[sel], status[sel]).coefs_[0])[0]
        if len(E):
            cox_intervals('split', X[held_out], time[held_out], status[held_out], E)

        for robust in [False, True]:
            E, intervals = randomized_inference(X, time, status, robust)
            if len(E):
                for method, (_, lower, upper) in intervals.items():
                    record(method, 'robust' if robust else 'model-based', E, lower, upper)

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py cox_misspecified`
and saved in `docs/results/cox_misspecified.csv`.

```{code-cell} ipython3
results = load_results('cox_misspecified')
display(provenance(results))
display(reproduce(results))

results['method'] = pd.Categorical(results['method'], ['naive', 'split', 'combination', 'carved'])
results['variance'] = pd.Categorical(results['variance'], ['model-based', 'robust'])
results['signal'] = pd.Categorical(results['signal'], ['strong', 'weak', 'null'])
(results.groupby(['method', 'variance'], observed=True)
        .agg(intervals=('covered', 'size'),
             coverage=('covered', 'mean'),
             mean_length=('length', 'mean'))
        .round(3))
```

By the strength of the signal, the coverage:

```{code-cell} ipython3
(results.groupby(['method', 'variance', 'signal'], observed=True)['covered'].mean()
        .unstack('signal').round(3))
```

DISCUSSION

(coxmis-refs)=
## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the
  Royal Statistical Society, Series B*, 34(2), 187–220.
- Efron, B. (1977). The efficiency of Cox's likelihood function for
  censored data. *Journal of the American Statistical Association*,
  72(359), 557–565.
- Struthers, C. A., & Kalbfleisch, J. D. (1986). Misspecified
  proportional hazard models. *Biometrika*, 73(2), 363–369.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox
  proportional hazards model. *Journal of the American Statistical
  Association*, 84(408), 1074–1078.
- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993).
  *Statistical Models Based on Counting Processes*. Springer.
- Lin, D. Y., Wei, L. J., & Ying, Z. (1993). Checking the Cox model with
  cumulative sums of martingale-based residuals. *Biometrika*, 80(3),
  557–572.
- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
