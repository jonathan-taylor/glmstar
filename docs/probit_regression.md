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

# Probit regression

The probit model for a binary response has
$\mu_i = P(y_i = 1) = \Phi(\eta_i)$, with
$\eta_i = \beta_0 + x_i^T\beta$ and $\Phi$ the standard normal
distribution function. Unlike the logit, the probit link is not the
canonical link of the binomial family, so the response does not enter
the log-likelihood only through $\sum_i y_i\,(1, x_i)$. This vignette
repeats the inference after selection of
[Score bootstraps with a linear term](score_bootstrap.md) for a
probit regression, and shows what changes when the link is not
canonical.

As there, we select with the lasso fit to the data with a random linear
term $-\omega$, which perturbs the score, and then either combine two
fits or carve with `lassoinf`: randomized selective inference
({ref}`Tian and Taylor, 2018 <probit-refs>`) with a parametric
randomization, an instance of data fission with resampled data
({ref}`Fry, Panigrahi and Taylor, 2025 <probit-refs>`).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl
from glmnet.inference import glmstar_inference_problem, score_perturbation
from lassoinf import LassoInference

probit = sm.families.Binomial(link=sm.families.links.Probit())
```

## Data

As in the logistic example, there are two strong signals, six weak ones
near the selection threshold, and 22 nulls. The features are correlated
and have standard deviations from 0.2 to 5, and the effects are set per
standard deviation of each feature. Probit coefficients are on a smaller
scale than logistic ones, so the effects are smaller.

```{code-cell} ipython3
rng = np.random.default_rng(3)
n, p, rho = 5000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.18, -0.15, 0.055, -0.045, 0.04, 0.03, -0.025, 0.02, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def sim_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    return X, rng.binomial(1, norm.cdf(-0.3 + X @ beta)).astype(float)

X, y = sim_data(n)
```

`probit_lasso` fits the lasso at one value of $\lambda$, with an
optional linear term `LinearTerm(coef=G, intercept=G_0)` adding
$G_0\beta_0 + G^T\beta$ to the objective. `relaxed_fit` refits a model
$E$ without a penalty, keeping the linear term. The randomized fit uses
tight convergence tolerances, as the conditional inference assumes the
lasso was solved exactly.

```{code-cell} ipython3
pi = 0.9
lam = 0.6 * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def probit_lasso(X, y, linear_term=None, control=None):
    return GLMNet(family=probit,
                  lambda_values=np.array([lam]),
                  linear_term=linear_term,
                  control=control or GLMNetControl()).fit(X, y)

def relaxed_fit(X, y, E, linear_term):
    others = sorted(set(range(X.shape[1])) - set(E))
    return GLMNet(family=probit,
                  lambda_values=np.array([0.]),
                  exclude=others,
                  linear_term=linear_term,
                  control=tight).fit(X, y).coefs_[0][E]
```

## The score of a non-canonical link

Write $h = \phi(\eta)$ for $d\mu/d\eta$, where $\phi$ is the standard
normal density, and $V(\mu) = \mu(1-\mu)$. The score of the probit
log-likelihood, scaled as `GLMNet` scales it, is

$$S(\beta) = \sum_i v_i\, a_i\,(y_i - \mu_i)\,(1, x_i), \qquad a_i = \frac{h_i}{V(\mu_i)},$$

with $v_i = 1/n$. For the logit $a_i = 1$. Here the weights $a_i$
depend on $\beta$, so a perturbation of the response is not a linear
term. Perturbing the response would also give up the binary data: with
a "response" $y'_i > 1$ the probit objective contains
$(y'_i - 1)\log(1 - \Phi(\eta_i))$, which tends to $-\infty$ like
$-\eta_i^2/2$, and the lasso has no solution for any $\lambda$.

So we perturb the score, with the weights frozen at a pilot fit
$\hat\eta$, the unpenalized probit regression on all $p$ features.
Draw two parametric replicates
$y^*_1, y^*_2 \sim \mathrm{Bernoulli}(\Phi(\hat\eta))$ and set

$$e = \sqrt{\frac{1-\pi}{\pi}}\;\frac{y^*_1 - y^*_2}{\sqrt 2}, \qquad \omega = \sum_i v_i\,\hat a_i\, e_i\,(1, x_i).$$

Given $\hat\eta$, $e_i$ has mean 0 and variance
$\frac{1-\pi}{\pi}\hat V_i$, so

$$\mathrm{Var}(\omega) = \frac{1-\pi}{\pi}\sum_i v_i^2\,\hat a_i^2\,\hat V_i\,(1, x_i)(1, x_i)^T = \frac{1-\pi}{\pi}\,\frac1n\,X_1^T V \mathrm{diag}\Big(\frac{\hat h^2}{\hat V}\Big) X_1,$$

with $X_1 = [\mathbf{1}, X]$ and $V = \mathrm{diag}(v)$. One factor of
$\hat V$ cancels, and what is left is $(1-\pi)/\pi$ times the Fisher
information, scaled as the covariance of the score. This is the relation
that makes selection with the linear term $-\omega$ imitate selection
on a random fraction $\pi$ of the data, as for the logit. Evaluating the
weights at the pilot fit rather than at the lasso solution only changes
$\omega$ at second order.

`score_perturbation` draws $-\omega$ given $\hat\eta$:

```{code-cell} ipython3
def probit_fit(X, y):
    return sm.GLM(y, sm.add_constant(X), family=probit).fit()

def pilot_eta(X, y):
    return sm.add_constant(X) @ probit_fit(X, y).params

def perturbation(X, eta_hat):
    return score_perturbation(X, eta_hat, probit, (1 - pi) / pi, rng=rng)
```

## Inference

Both ways of doing inference from the logistic example carry over.

- **Combination.** $T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$,
  where $\hat\beta_E$ is the probit fit of $y$ on $X_E$ and
  $\hat\beta^*_E$ its fit with the linear term $-\omega$. Its
  standard errors are the probit ones divided by $\sqrt{1-\pi}$.
- **Carving.** The selection problem is the lasso problem for $y$ with
  the linear term $-\omega$. `glmstar_inference_problem` builds it from
  the randomized fit, with the probit score and, for `Q_hat`, the Fisher
  information $X_1^TV\mathrm{diag}(h^2/V)X_1$. For a non-canonical link
  this differs from the Hessian of the loss by a term with mean 0. It is
  evaluated at the one-step relaxed fit on the selected model, and
  `scalar_noise = (1 - pi) / pi`.

`randomized_inference` does one randomized selection and returns both
sets of intervals. The selected model $E$ is the one `lassoinf`
conditions on. It counts a coefficient within $10^{-6}$ of 0 as 0, so
on rare occasions it drops a variable that the solver left just inside
the active set, with a coefficient of order $10^{-7}$. Both methods use
this $E$.

```{code-cell} ipython3
def randomized_inference(X, y):
    omega = perturbation(X, pilot_eta(X, y))
    fit = probit_lasso(X, y, linear_term=omega, control=tight)
    if not np.any(fit.coefs_[0]):
        return [], None

    # carving
    info = glmstar_inference_problem(fit, X, y, scalar_noise=(1 - pi) / pi)
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_
    summary = summary.loc[summary.index > 0]       # row 0 is the intercept
    E = summary.index.values - 1
    if len(E) == 0:
        return E, None
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    full = probit_fit(X[:, E], y)
    T = (full.params[1:] - pi * relaxed_fit(X, y, E, omega)) / (1 - pi)
    se = full.bse[1:] / np.sqrt(1 - pi)
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the probit regression of $y$ on
$X_E$ in the population. We approximate them by fitting each selected
model to one large sample:

```{code-cell} ipython3
pop_X, pop_y = sim_data(200000)
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        targets[key] = probit_fit(pop_X[:, E], pop_y).params[1:]
    return targets[key]
```

### One data set

```{code-cell} ipython3
E, intervals = randomized_inference(X, y)
table = pd.concat({m: pd.DataFrame({'estimate': est, 'lower': lower, 'upper': upper},
                                   index=[f'X{j}' for j in E])
                   for m, (est, lower, upper) in intervals.items()}, axis=1)
table['target'] = target(E)
table.round(3)
```

## Calibration

We repeat the experiment and compare four methods:

- **naive**: select with the lasso on the data, then use the usual
  probit intervals from the same data;
- **split**: select on a random 90% of the data, then form intervals on
  the other 10%;
- **combination** and **carved**: select with the linear term
  $-\omega$, as above.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length per standard
deviation of the feature.

```{code-cell} ipython3
rows = []

def record(method, E, lower, upper):
    for j, lo, hi, t in zip(E, lower, upper, target(E)):
        rows.append({'method': method, 'signal': signal[j],
                     'covered': lo <= t <= hi, 'length': (hi - lo) * scale[j]})

def probit_intervals(X, y, E):
    fit = probit_fit(X[:, E], y)
    return fit.params[1:] - q * fit.bse[1:], fit.params[1:] + q * fit.bse[1:]

for _ in range(300):
    X, y = sim_data(n)

    E = np.nonzero(probit_lasso(X, y).coefs_[0])[0]
    if len(E):
        record('naive', E, *probit_intervals(X, y, E))

    idx = rng.permutation(n)
    sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
    E = np.nonzero(probit_lasso(X[sel], y[sel]).coefs_[0])[0]
    if len(E):
        record('split', E, *probit_intervals(X[held_out], y[held_out], E))

    E, intervals = randomized_inference(X, y)
    if len(E):
        for method, (_, lower, upper) in intervals.items():
            record(method, E, lower, upper)

methods = ['naive', 'split', 'combination', 'carved']
results = pd.DataFrame(rows)
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
for inference. The split, combination and carved intervals are
calibrated for each kind of signal. As for the logit, the split and
combination intervals have about the same length, $1/\sqrt{1-\pi}$ times
the full-data standard error, and the carved intervals are much
shorter. For the strong signals they are close to the naive ones, as
these variables would be selected for almost any $\omega$. For the weak
signals and the nulls they widen, by the cost of conditioning on a
selection that might easily have gone the other way, but remain shorter
than the split and combination intervals.

The calibration relies on the probit model: $e$ has the variance of $y$
under the fitted model, and both the cancellation in $\mathrm{Var}(\omega)$
and the variance `lassoinf` uses for the score are the probit Fisher
information. Without the model, the randomization can instead come
from a nonparametric resample of the observations
({ref}`Fry, Panigrahi and Taylor, 2025 <probit-refs>`).

(probit-refs)=
## References

- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
