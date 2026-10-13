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

# Score bootstraps with a linear term

`GLMNet` takes a `linear_term` $G$, one entry per feature, and adds
$G^T\beta$ to the objective:

$$-\frac{1}{\sum_i w_i}\sum_i w_i\,\ell_i(\beta_0 + x_i^T\beta) \;+\; G^T\beta \;+\; \lambda\,P_\alpha(\beta),$$

where $\ell_i$ is the log-likelihood of observation $i$, $w_i$ its
weight and $\beta$ the coefficients on the original scale of the
features. The penalty $P_\alpha$ is applied as usual, and the design is
standardized as usual if `standardize=True`.

For the GLMs here the data enter this objective only through the score
$\sum_i v_i\, x_i\, y_i$, where $v_i = w_i/\sum_j w_j$ are the normalized
weights. Changing the response is therefore the same as changing $G$.
This means a bootstrap can perturb the score directly, drawing a
$p$-vector instead of a new response vector of length $n$, and refit
each replicate on the original data. This is the idea behind
resampling the estimating equations ({ref}`Parzen, Wei and Ying, 1994 <score-bootstrap-refs>`)
and the score bootstrap ({ref}`Kline and Santos, 2012 <score-bootstrap-refs>`).

Below we use it first for a parametric bootstrap of the lasso in
weighted least squares. For logistic regression we then perturb the
score with parametric replicates, in a way that imitates data
splitting. This gives valid inference after the lasso has selected the
model, without holding any data out.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl
```

## Weighted least squares

We simulate heteroskedastic data with known weights. The features are on
quite different scales, so the standardization matters.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 300, 8
scale = rng.uniform(0.5, 3, p)
X = rng.standard_normal((n, p)) * scale
beta = np.r_[1, -0.75, 0.5, np.zeros(p - 3)] / scale
w = rng.uniform(0.5, 2, n)
y = 1 + X @ beta + rng.standard_normal(n) / np.sqrt(w)
D = pd.DataFrame({'y': y, 'w': w})
v = w / w.sum()
```

The parametric bootstrap starts from the weighted least squares fit,
as R's `lm(y ~ X, weights=w)`. That gives $\hat\beta$ on the original
scale and $\hat\sigma^2 = \sum_i w_i \hat r_i^2/(n-p-1)$. A replicate is

$$y^* = \hat\beta_0 + X\hat\beta + \epsilon^*, \qquad \epsilon^*_i \sim N(0, \hat\sigma^2/w_i).$$

```{code-cell} ipython3
X1 = np.column_stack([np.ones(n), X])
sw = np.sqrt(w)
coef_ls = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)[0]
fitted = X1 @ coef_ls
resid = y - fitted
sigma2 = (w * resid**2).sum() / (n - p - 1)
sigma2
```

We bootstrap the lasso at one value of $\lambda$, taken from the path
fit to the data.

```{code-cell} ipython3
path = GLMNet(response_id='y', weight_id='w').fit(X, D)
lam = path.lambda_values_[20]
print(lam, (path.coefs_[20] != 0).sum())

def lasso(D, linear_term=None):
    fit = GLMNet(response_id='y', weight_id='w',
                 lambda_values=np.array([lam]),
                 linear_term=linear_term).fit(X, D)
    return fit.coefs_[0], fit.intercepts_[0]
```

### The replicate as a linear term

The least squares residuals satisfy $\sum_i v_i \hat r_i = 0$ and
$X^TV\hat r = 0$, with $V = \mathrm{diag}(v)$. Expanding the square, the
lasso objective for $y^*$ is the objective for $y + \bar\epsilon^*$
with the linear term

$$G^* = -\tilde X^T V \epsilon^*, \qquad \tilde X = X - \mathbf{1}\bar x^T, \quad \bar x = X^T v, \quad \bar\epsilon^* = \textstyle\sum_i v_i \epsilon^*_i,$$

up to a constant. Shifting the response by the constant $\bar\epsilon^*$
only shifts the (unpenalized) intercept. So the bootstrap coefficients
are those of the original data with linear term $G^*$, and the
intercept is shifted by $\bar\epsilon^*$:

```{code-cell} ipython3
Xc = X - v @ X
eps = rng.standard_normal(n) * np.sqrt(sigma2 / w)

coef_star, int_star = lasso(pd.DataFrame({'y': fitted + eps, 'w': w}))
coef_lin, int_lin = lasso(D, linear_term=-Xc.T @ (v * eps))

print(np.abs(coef_star - coef_lin).max(),
      np.abs(int_star - (int_lin + (v * eps).sum())).max())
```

The two agree even though `GLMNet` standardizes the design. Both
objectives are written in terms of $\beta$ on the original scale, and
`GLMNet` converts the linear term to the standardized coordinates
itself.

### The score bootstrap

The replicate enters only through the $p$-vector
$U^* = \tilde X^T V\epsilon^*$. Since $v_i^2/w_i = v_i/\sum_j w_j$,

$$U^* \sim N\Big(0,\ \frac{\hat\sigma^2}{\sum_j w_j}\,\tilde X^T V \tilde X\Big),$$

so we can draw $U^*$ directly and fit with `linear_term=-U*`. For
weighted least squares $U^*$ is exactly Gaussian, so this is exactly
the parametric bootstrap, not an approximation to it.

The wild score bootstrap of Kline and Santos instead uses
$U^* = \tilde X^T V(\hat r \circ \xi)$ with independent Rademacher signs
$\xi_i$. This keeps the heteroskedasticity of the residuals instead of
assuming $\mathrm{Var}(y_i) = \sigma^2/w_i$. Only the draw of $U^*$
changes.

```{code-cell} ipython3
B = 200
cov_U = sigma2 / w.sum() * Xc.T @ (v[:, None] * Xc)
L = np.linalg.cholesky(cov_U)

boot = {'refit y*': [], 'score': [], 'wild score': []}
for _ in range(B):
    eps = rng.standard_normal(n) * np.sqrt(sigma2 / w)
    boot['refit y*'].append(lasso(pd.DataFrame({'y': fitted + eps, 'w': w}))[0])
    U = L @ rng.standard_normal(p)
    boot['score'].append(lasso(D, linear_term=-U)[0])
    xi = rng.choice([-1., 1.], n)
    boot['wild score'].append(lasso(D, linear_term=-Xc.T @ (v * resid * xi))[0])
boot = {k: np.array(b) for k, b in boot.items()}
```

The three bootstraps agree, up to Monte Carlo error. The table shows the
standard deviation of each coefficient and how often it is selected:

```{code-cell} ipython3
summary = pd.concat({k: pd.DataFrame({'sd': b.std(0),
                                      'P(selected)': (b != 0).mean(0)})
                     for k, b in boot.items()}, axis=1)
summary.index = [f'X{j}' for j in range(p)]
summary.round(3)
```

## Logistic regression

### The response enters only through the score

The logistic log-likelihood $y\,\eta - \log(1 + e^\eta)$ is linear in
$y$. So fitting the lasso to any response $y'$ is the same as fitting it
to another response $\tilde y$ with the linear term

$$G = -X^T V (y' - \tilde y),$$

provided $\sum_i v_i \tilde y_i = \sum_i v_i y'_i$ (this keeps the
intercept's part of the score). The simplest choice is the constant
$\tilde y = \sum_i v_i y'_i$. Then $y'$ need not be binary, or even lie in
$[0, 1]$: only its score is used.

We simulate from a logistic model with correlated features and a few
weak signals, as in a typical selection problem:

```{code-cell} ipython3
rng = np.random.default_rng(1)
n, p, rho = 5000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
beta = np.r_[0.135, -0.105, 0.09, 0.075, -0.06, 0.045, np.zeros(p - 6)]

def sim_data(n):
    X = rng.standard_normal((n, p)) @ R
    mu = 1 / (1 + np.exp(-(-0.5 + X @ beta)))
    return X, rng.binomial(1, mu).astype(float)

X, y = sim_data(n)
v = np.ones(n) / n
```

`logistic_lasso` fits the lasso at one value of $\lambda$, on the
average-loss scale `GLMNet` uses. With `relax=True` it also refits the
selected model without a penalty, as R's `relax.glmnet` does.
`randomized_lasso` fits the lasso to a response $y'$ by way of the
constant response and the linear term:

```{code-cell} ipython3
pi = 0.9
lam = 0.5 * np.sqrt(2 * np.log(p) / (pi * n))

def logistic_lasso(X, response, linear_term=None, relax=False, control=None):
    with warnings.catch_warnings():
        # a constant response has null deviance 0: its deviance ratio is undefined
        warnings.simplefilter('ignore', RuntimeWarning)
        return GLMNet(family=sm.families.Binomial(),
                      lambda_values=np.array([lam]),
                      linear_term=linear_term,
                      relax=relax,
                      control=control or GLMNetControl()).fit(X, response)

def randomized_lasso(y_prime, relax=False, control=None):
    y_tilde = (v * y_prime).sum()
    return logistic_lasso(X, np.full(n, y_tilde),
                          linear_term=-X.T @ (v * (y_prime - y_tilde)),
                          relax=relax, control=control)
```

With tight convergence tolerances, the two ways of fitting a binary
response agree to rounding error:

```{code-cell} ipython3
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)
direct = logistic_lasso(X, y, control=tight).coefs_[0]
via_linear_term = randomized_lasso(y, control=tight).coefs_[0]
np.abs(direct - via_linear_term).max()
```

### A parametric score perturbation

Let $\hat\mu$ be the fitted means of the unpenalized logistic regression
on all $p$ features. Draw two independent parametric replicates
$y^*_1, y^*_2 \sim \mathrm{Bernoulli}(\hat\mu)$ and set

$$\omega = \sqrt{\frac{1-\pi}{\pi}}\;\frac{y^*_1 - y^*_2}{\sqrt 2}, \qquad 0 < \pi < 1.$$

Given $\hat\mu$, $\omega_i$ has mean 0 and variance
$\frac{1-\pi}{\pi}\,\hat\mu_i(1-\hat\mu_i)$. This is the Bernoulli
variance of $y_i$ under the fitted model, scaled by $(1-\pi)/\pi$, with
no Gaussian approximation. Taking the difference of two replicates
centers $\omega$ exactly.

We select variables with the lasso fit to $y + \omega$, using all $n$
observations. Its score is the data's score plus the perturbation
$X^TV\omega$, including the intercept's part $\sum_i v_i\omega_i$.

```{code-cell} ipython3
def fitted_means(X, y):
    return sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit().fittedvalues

def omega(mu_hat):
    y1, y2 = rng.binomial(1, mu_hat), rng.binomial(1, mu_hat)
    return np.sqrt((1 - pi) / pi) * (y1 - y2) / np.sqrt(2)
```

### How this imitates data splitting

Write $S$ for the score of the full data, scaled as `GLMNet` scales it
(an average over observations). Data splitting selects with the score
$S_I$ of a random fraction $\pi$ of the observations. Over the random
split, $\mathrm{Cov}(S_I, S) = \mathrm{Var}(S)$ and
$\mathrm{Var}(S_I) \approx \mathrm{Var}(S)/\pi$. So

$$S_I = S + \omega_I, \qquad \omega_I \text{ uncorrelated with } S, \quad \mathrm{Var}(\omega_I) \approx \frac{1-\pi}{\pi}\,\mathrm{Var}(S).$$

To first order, the lasso on the selection part of the split is the
lasso on all the data with the linear term $-\omega_I$. Under the
logistic model, the perturbation $X^TV\omega$ above has (given
$\hat\mu$) the same covariance as $\omega_I$. Selecting with $y + \omega$
therefore behaves like selecting on a random $\pi$-fraction of the data.

Splitting then does inference on the held-out fraction $1-\pi$. Its
score is, to first order, $S - \frac{\pi}{1-\pi}\,\omega_I$. The
randomized version uses the same combination with $\omega$ in place of
$\omega_I$. This combination is uncorrelated with the selection score:

$$\mathrm{Cov}\Big(S + \omega,\; S - \tfrac{\pi}{1-\pi}\,\omega\Big) = \mathrm{Var}(S) - \tfrac{\pi}{1-\pi}\cdot\tfrac{1-\pi}{\pi}\,\mathrm{Var}(S) = 0,$$

and its variance is $\mathrm{Var}(S)/(1-\pi)$. Asymptotically both are
Gaussian, so it is independent of the selection.

Linearizing the unpenalized estimators of a selected model $E$ turns
this into a statement about estimators. Let $\hat\beta_E$ be the fit of
model $E$ to $y$ and $\hat\beta^*_E$ its fit to $y + \omega$. Then

$$T_E = \hat\beta_E - \frac{\pi}{1-\pi}\big(\hat\beta^*_E - \hat\beta_E\big) = \frac{\hat\beta_E - \pi\,\hat\beta^*_E}{1-\pi} \;\approx\; N\Big(\beta_E,\ \frac{\mathrm{Var}(\hat\beta_E)}{1-\pi}\Big),$$

independent of the selection. Here $\beta_E$ is the projection
parameter, the population logistic regression of $y$ on $X_E$. The
interval is as wide as one computed on a held-out fraction $1-\pi$, but
no observations are held out.

This parametric score bootstrap is an example of data fission with
resampled data
({ref}`Fry, Panigrahi and Taylor, 2025 <score-bootstrap-refs>`). There
the randomization is a resample of the data, such as a 9n-of-n
bootstrap, which needs no model; here it is drawn from the fitted
logistic model. More generally, it is an instance of randomized
selective inference
({ref}`Tian and Taylor, 2018 <score-bootstrap-refs>`).

### The recipe

1. Draw $\omega$, and fit the lasso to $y + \omega$ with `relax=True`.
   Its selected set is $E$. Its relaxed coefficients are
   $\hat\beta^*_E$, the unpenalized refit on $E$ with the same
   randomized score, because the refit carries over the linear term.
2. Fit the logistic regression of $y$ on $X_E$: this gives $\hat\beta_E$
   and its standard errors.
3. Report $T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$, with standard
   errors $\widehat{\mathrm{se}}(\hat\beta_E)/\sqrt{1-\pi}$.

```{code-cell} ipython3
def logit_fit(X, y):
    fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    return fit.params[1:], fit.bse[1:]

def randomized_inference(X, y):
    fit = randomized_lasso(y + omega(fitted_means(X, y)), relax=True)
    E = np.nonzero(fit.coefs_[0])[0]
    if len(E) == 0:
        return E, None, None
    beta_star = fit.relaxed_coefs_[0][E]
    beta_hat, se = logit_fit(X[:, E], y)
    return E, (beta_hat - pi * beta_star) / (1 - pi), se / np.sqrt(1 - pi)
```

The targets $\beta_E$ are projections. We approximate them by fitting
each selected model to one large sample from the population:

```{code-cell} ipython3
pop_X, pop_y = sim_data(200000)
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        targets[key] = logit_fit(pop_X[:, E], pop_y)[0]
    return targets[key]
```

### One data set

```{code-cell} ipython3
q = norm.ppf(0.95)  # nominal 90% intervals
E, T, se = randomized_inference(X, y)
pd.DataFrame({'estimate': T, 'se': se,
              'lower': T - q * se, 'upper': T + q * se,
              'target': target(E)},
             index=[f'X{j}' for j in E]).round(3)
```

### Calibration

We repeat the experiment and compare three methods:

- **naive**: select with the lasso on the data, then use the usual
  intervals from the same data;
- **split**: select on a random 90% of the data, then form intervals on
  the other 10%;
- **randomized**: the recipe above, with $\pi = 0.9$.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, the interval's length, and the
$z$-statistic $(\text{estimate} - \beta_{E,j})/\text{se}$. The column
`sd_z` is the standard deviation of the $z$-statistics. It is about 1
when the standard errors are honest, and above 1 when they are too
small.

```{code-cell} ipython3
def one_rep():
    global X, y
    X, y = sim_data(n)
    out = {}

    E = np.nonzero(logistic_lasso(X, y).coefs_[0])[0]
    if len(E):
        b, se = logit_fit(X[:, E], y)
        out['naive'] = ((b - target(E)) / se, se)

    idx = rng.permutation(n)
    sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
    E = np.nonzero(logistic_lasso(X[sel], y[sel]).coefs_[0])[0]
    if len(E):
        b, se = logit_fit(X[held_out][:, E], y[held_out])
        out['split'] = ((b - target(E)) / se, se)

    E, T, se = randomized_inference(X, y)
    if len(E):
        out['randomized'] = ((T - target(E)) / se, se)
    return out

reps = [one_rep() for _ in range(300)]
z = {m: np.concatenate([r[m][0] for r in reps if m in r]) for m in ['naive', 'split', 'randomized']}
se = {m: np.concatenate([r[m][1] for r in reps if m in r]) for m in z}

pd.DataFrame({'n_intervals': {m: len(z[m]) for m in z},
              'coverage_90': {m: np.mean(np.fabs(z[m]) < q) for m in z},
              'sd_z': {m: z[m].std() for m in z},
              'mean_length_90': {m: (2 * q * se[m]).mean() for m in z}}).round(3)
```

If inference is calibrated, the $z$-statistics are $N(0,1)$, so the
two-sided $p$-values $2\{1-\Phi(|z|)\}$ are uniform:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, m in zip(axes, z):
    pvalues = np.sort(2 * norm.sf(np.fabs(z[m])))
    ax.plot((np.arange(len(pvalues)) + 0.5) / len(pvalues), pvalues, '.', ms=3)
    ax.plot([0, 1], [0, 1], 'r')
    ax.set_title(m)
    ax.set_xlabel('uniform quantiles')
axes[0].set_ylabel('p-values');
```

The naive intervals undercover, because the variables were chosen with
the same data used for inference. Splitting and the randomized fits are
both calibrated, with intervals of the same width: $1/\sqrt{1-\pi}$
times the full-data standard error. But the randomized fits hold no
data out. The parametric perturbation $\omega$ plays the role of the
random split, for both the selection and the inference.

The calibration relies on the logistic model. $\omega$ has the
variance of $y$ under the fitted model, and the cancellation in the
covariance above needs that to match the true variance of the score.
Without the model, the randomization can instead come from a
nonparametric bootstrap of the observations, as in Fry, Panigrahi and
Taylor (2025).

(score-bootstrap-refs)=
## References

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
