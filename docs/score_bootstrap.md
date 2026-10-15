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

`GLMNet` takes a `linear_term`, a `LinearTerm(coef=G, intercept=G_0)`
with one entry per feature and one for the intercept, and adds
$G_0\beta_0 + G^T\beta$ to the objective:

$$-\frac{1}{\sum_i w_i}\sum_i w_i\,\ell_i(\beta_0 + x_i^T\beta) \;+\; G_0\beta_0 + G^T\beta \;+\; \lambda\,P_\alpha(\beta),$$

where $\ell_i$ is the log-likelihood of observation $i$, $w_i$ its
weight and $\beta_0, \beta$ the intercept and coefficients on the
original scale of the features. The penalty $P_\alpha$ is applied as
usual, and the design is standardized as usual if `standardize=True`.

For the GLMs here, with their canonical links, the data enter this
objective only through the score $\sum_i v_i\, y_i\,(1, x_i)$, where
$v_i = w_i/\sum_j w_j$ are the normalized weights. Changing the response
is therefore the same as changing $(G_0, G)$.
This means a bootstrap can perturb the score directly, drawing a
$p$-vector instead of a new response vector of length $n$, and refit
each replicate on the original data. This is the idea behind
resampling the estimating equations ({ref}`Parzen, Wei and Ying, 1994 <score-bootstrap-refs>`)
and the score bootstrap ({ref}`Kline and Santos, 2012 <score-bootstrap-refs>`).

Below we use it first for a parametric bootstrap of the lasso in
weighted least squares. For logistic regression we then perturb the
score with parametric replicates, in a way that imitates data
splitting. This gives valid inference after the lasso has selected the
model, without holding any data out, either by combining two fits or by
carving with `lassoinf`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from glmnet import GLMNet, LinearTerm
from glmnet.glmnet import GLMNetControl
from glmnet.inference import glmstar_inference_problem, score_perturbation
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
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
lasso objective for $y^*$ is the objective for $y$ with the linear term

$$G^* = -X^T V \epsilon^*, \qquad G^*_0 = -\textstyle\sum_i v_i \epsilon^*_i,$$

up to a constant. So the bootstrap fit, intercept included, is the fit
to the original data with this linear term:

```{code-cell} ipython3
eps = rng.standard_normal(n) * np.sqrt(sigma2 / w)

coef_star, int_star = lasso(pd.DataFrame({'y': fitted + eps, 'w': w}))
coef_lin, int_lin = lasso(D, linear_term=LinearTerm(coef=-X.T @ (v * eps),
                                                    intercept=-(v * eps).sum()))

print(np.abs(coef_star - coef_lin).max(), np.abs(int_star - int_lin))
```

The two agree even though `GLMNet` standardizes the design. Both
objectives are written in terms of $\beta$ on the original scale, and
`GLMNet` converts the linear term to the standardized coordinates
itself.

### The score bootstrap

The replicate enters only through the $(p+1)$-vector
$U^* = X_1^T V\epsilon^*$, with $X_1 = [\mathbf{1}, X]$. Since
$v_i^2/w_i = v_i/\sum_j w_j$,

$$U^* \sim N\Big(0,\ \frac{\hat\sigma^2}{\sum_j w_j}\,X_1^T V X_1\Big),$$

so we can draw $U^*$ directly and fit with the linear term $-U^*$, its
first entry the intercept's. For weighted least squares $U^*$ is
exactly Gaussian, so this is exactly the parametric bootstrap, not an
approximation to it.

The wild score bootstrap of Kline and Santos instead uses
$U^* = X_1^T V(\hat r \circ \xi)$ with independent Rademacher signs
$\xi_i$. This keeps the heteroskedasticity of the residuals instead of
assuming $\mathrm{Var}(y_i) = \sigma^2/w_i$. Only the draw of $U^*$
changes.

```{code-cell} ipython3
B = 200
cov_U = sigma2 / w.sum() * X1.T @ (v[:, None] * X1)
L = np.linalg.cholesky(cov_U)

def minus(U):
    """The linear term -U, for U = (intercept entry, coefficient entries)."""
    return LinearTerm(coef=-U[1:], intercept=-U[0])

boot = {'refit y*': [], 'score': [], 'wild score': []}
for _ in range(B):
    eps = rng.standard_normal(n) * np.sqrt(sigma2 / w)
    boot['refit y*'].append(lasso(pd.DataFrame({'y': fitted + eps, 'w': w}))[0])
    U = L @ rng.standard_normal(p + 1)
    boot['score'].append(lasso(D, linear_term=minus(U))[0])
    xi = rng.choice([-1., 1.], n)
    boot['wild score'].append(lasso(D, linear_term=minus(X1.T @ (v * resid * xi)))[0])
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

### Randomizing the score, not the response

We randomize the selection by perturbing the score: the lasso is fit
to the data $y$ with the linear term $-\omega$, for a random
$(p+1)$-vector $\omega$. The response is never changed, so it stays
binary.

The logistic log-likelihood $y\,\eta - \log(1 + e^\eta)$ is linear in
$y$, so a perturbation $e$ of the response would change the score by
exactly $\omega = \sum_i v_i e_i\,(1, x_i)$. For a link that is not
canonical the score is $\sum_i v_i a_i (y_i - \mu_i)(1, x_i)$ with
$a = h/V$ ($h = d\mu/d\eta$, $V$ the variance function), and
`score_perturbation` freezes the weights $a$ at a pilot fit. Either
way the randomization is a fixed linear term, and the objective
stays that of the binary data.

We simulate from a logistic model with correlated features: two strong
signals, six weak ones near the selection threshold, and 22 nulls. The
features have standard deviations from 0.2 to 5, so `GLMNet`'s
standardization matters: the linear term, the score perturbation and the
carving all have to be put on the right scale. The effects are set per
standard deviation of each feature.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n, p, rho = 5000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.3, -0.25, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def sim_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    mu = 1 / (1 + np.exp(-(-0.5 + X @ beta)))
    return X, rng.binomial(1, mu).astype(float)

X, y = sim_data(n)
```

`logistic_lasso` fits the lasso at one value of $\lambda$, on the
average-loss scale `GLMNet` uses. With `relax=True` it also refits the
selected model without a penalty, as R's `relax.glmnet` does; the
refit keeps the linear term. The randomized fit uses tight convergence
tolerances, as the conditional inference below assumes the lasso was
solved exactly.

```{code-cell} ipython3
pi = 0.9
lam = 0.5 * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals

tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def logistic_lasso(X, y, linear_term=None, relax=False, control=None):
    return GLMNet(family=sm.families.Binomial(),
                  lambda_values=np.array([lam]),
                  linear_term=linear_term,
                  relax=relax,
                  control=control or GLMNetControl()).fit(X, y)
```

### A parametric score perturbation

Let $\hat\mu$ be the fitted means of the unpenalized logistic regression
on all $p$ features. Draw two independent parametric replicates
$y^*_1, y^*_2 \sim \mathrm{Bernoulli}(\hat\mu)$ and set

$$e = \sqrt{\frac{1-\pi}{\pi}}\;\frac{y^*_1 - y^*_2}{\sqrt 2}, \qquad \omega = \sum_i v_i\, e_i\,(1, x_i), \qquad 0 < \pi < 1.$$

Given $\hat\mu$, $e_i$ has mean 0 and variance
$\frac{1-\pi}{\pi}\,\hat\mu_i(1-\hat\mu_i)$. This is the Bernoulli
variance of $y_i$ under the fitted model, scaled by $(1-\pi)/\pi$, with
no Gaussian approximation. Taking the difference of two replicates
centers $e$ exactly. So $\omega$ has $(1-\pi)/\pi$ times the covariance
of the score, including the intercept's part $\sum_i v_i e_i$.

We select variables with the lasso fit to $y$ with the linear term
$-\omega$, using all $n$ observations. `score_perturbation` draws it,
given the pilot linear predictor $\hat\eta$:

```{code-cell} ipython3
def pilot_eta(X, y):
    X1 = sm.add_constant(X)
    return X1 @ sm.GLM(y, X1, family=sm.families.Binomial()).fit().params

def perturbation(X, eta_hat):
    return score_perturbation(X, eta_hat, sm.families.Binomial(), (1 - pi) / pi, rng=rng)
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
logistic model, the perturbation $\omega$ above has (given
$\hat\mu$) the same covariance as $\omega_I$. Selecting with the linear
term $-\omega$ therefore behaves like selecting on a random
$\pi$-fraction of the data.

Splitting then does inference on the held-out fraction $1-\pi$. Its
score is, to first order, $S - \frac{\pi}{1-\pi}\,\omega_I$. The
randomized version uses the same combination with $\omega$ in place of
$\omega_I$. This combination is uncorrelated with the selection score:

$$\mathrm{Cov}\Big(S + \omega,\; S - \tfrac{\pi}{1-\pi}\,\omega\Big) = \mathrm{Var}(S) - \tfrac{\pi}{1-\pi}\cdot\tfrac{1-\pi}{\pi}\,\mathrm{Var}(S) = 0,$$

and its variance is $\mathrm{Var}(S)/(1-\pi)$. Asymptotically both are
Gaussian, so it is independent of the selection.

Linearizing the unpenalized estimators of a selected model $E$ turns
this into a statement about estimators. Let $\hat\beta_E$ be the fit of
model $E$ to $y$ and $\hat\beta^*_E$ its fit to $y$ with the linear term
$-\omega$. Then

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
logistic model.

### Carving without holding out data

$T_E$ is independent of the selection because it keeps only the part of
the score that is uncorrelated with the selection score $S + \omega$. The
price is that its variance is $1/(1-\pi)$ times that of the full-data
estimator, ten times for $\pi = 0.9$, however strong the signal.

Carving keeps all of $S$ and instead conditions on the selection: the
intervals come from the distribution of the full-data estimator given
that the lasso chose $E$. With a split, the randomization is
$\omega_I = S_I - S$ and `lassoinf` handles it with
`scalar_noise` $= (1-\pi)/\pi$ (see
[Selective inference after a logistic lasso](inference.md)). The
parametric perturbation has the same structure. The selection score is
$S$ plus a perturbation that, given $\hat\mu$, is independent of $S$
with $(1-\pi)/\pi$ times its variance. So the same conditional inference
applies, with the same `scalar_noise`, and no rows are removed or
resampled:

- the selection problem is the lasso problem for $y$ with the linear
  term $-\omega$, which `glmstar_inference_problem` builds from the
  randomized fit, linear term included;
- with `scalar_noise=(1 - pi) / pi`, it linearizes the full-data score
  $S$ of $y$ (without the linear term) at the selected model, as it
  does when carving with a split.

This is randomized selective inference
({ref}`Tian and Taylor, 2018 <score-bootstrap-refs>`) with the
parametric perturbation as the randomization. How much the selection
costs depends on how surprising it was. A strong signal would be
selected for almost any $\omega$, so conditioning on its selection
changes little: its interval should be close to the full-data one. A
variable near the selection threshold pays more, but it still uses all
of $S$.

### The recipe

1. Draw $\omega$ and fit the lasso to $y$ with `linear_term=-omega`
   and `relax=True`. Its selected set is $E$. Its relaxed coefficients
   are $\hat\beta^*_E$, the unpenalized refit on $E$ with the same
   randomized score, because the refit carries over the linear term.
2. **Combination.** Fit the logistic regression of $y$ on $X_E$, giving
   $\hat\beta_E$ and its standard errors. Report
   $T_E = (\hat\beta_E - \pi\hat\beta^*_E)/(1-\pi)$, with standard
   errors $\widehat{\mathrm{se}}(\hat\beta_E)/\sqrt{1-\pi}$.
3. **Carving.** Pass the randomized fit and the data $(X, y)$ to
   `glmstar_inference_problem`, and its output to `lassoinf`.

```{code-cell} ipython3
def logit_fit(X, y):
    fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    return fit.params[1:], fit.bse[1:]

def randomized_selection(X, y):
    omega = perturbation(X, pilot_eta(X, y))
    fit = logistic_lasso(X, y, linear_term=omega, relax=True, control=tight)
    return fit, np.nonzero(fit.coefs_[0])[0]

def combination_intervals(fit, E, X, y):
    beta_star = fit.relaxed_coefs_[0][E]
    beta_hat, se = logit_fit(X[:, E], y)
    T = (beta_hat - pi * beta_star) / (1 - pi)
    se = se / np.sqrt(1 - pi)
    return T, T - q * se, T + q * se

def carved_intervals(fit, E, X, y):
    info = glmstar_inference_problem(fit, X, y, scalar_noise=(1 - pi) / pi)
    # row 0 is the intercept
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_.loc[E + 1]
    return (summary['beta_hat'].values, summary['lower_conf'].values,
            summary['upper_conf'].values)
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
fit, E = randomized_selection(X, y)
intervals = {'combination': combination_intervals(fit, E, X, y),
             'carved': carved_intervals(fit, E, X, y)}
table = pd.concat({m: pd.DataFrame({'estimate': est, 'lower': lower, 'upper': upper},
                                   index=[f'X{j}' for j in E])
                   for m, (est, lower, upper) in intervals.items()}, axis=1)
table['target'] = target(E)
table.round(3)
```

The combination estimates are noisy: dividing by $1-\pi$ multiplies
the noise in $\hat\beta_E - \pi\hat\beta^*_E$ by 10. The carved
intervals, which use all of $S$, are about a third as long.

### Calibration

We repeat the experiment and compare four methods:

- **naive**: select with the lasso on the data, then use the usual
  intervals from the same data;
- **split**: select on a random 90% of the data, then form intervals on
  the other 10%;
- **combination**: select with the linear term $-\omega$, then use
  $T_E$;
- **carved**: select with the linear term $-\omega$, then condition on
  the selection.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length. The length is
multiplied by the feature's standard deviation, so it is the length for
the effect of one standard deviation and comparable across features.

```{code-cell} ipython3
def calibration(n_rep=300):
    rows = []

    def record(method, E, lower, upper):
        for j, lo, hi, t in zip(E, lower, upper, target(E)):
            rows.append({'method': method, 'signal': signal[j],
                         'covered': lo <= t <= hi, 'length': (hi - lo) * scale[j]})

    for _ in range(n_rep):
        X, y = sim_data(n)

        E = np.nonzero(logistic_lasso(X, y).coefs_[0])[0]
        if len(E):
            b, se = logit_fit(X[:, E], y)
            record('naive', E, b - q * se, b + q * se)

        idx = rng.permutation(n)
        sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
        E = np.nonzero(logistic_lasso(X[sel], y[sel]).coefs_[0])[0]
        if len(E):
            b, se = logit_fit(X[held_out][:, E], y[held_out])
            record('split', E, b - q * se, b + q * se)

        fit, E = randomized_selection(X, y)
        if len(E):
            record('combination', E, *combination_intervals(fit, E, X, y)[1:])
            record('carved', E, *carved_intervals(fit, E, X, y)[1:])

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py score_bootstrap`
and saved in `docs/results/score_bootstrap.csv`.

```{code-cell} ipython3
results = load_results('score_bootstrap')
display(provenance(results))
display(reproduce(results))

results['method'] = pd.Categorical(results['method'],
                                   ['naive', 'split', 'combination', 'carved'])
results['signal'] = pd.Categorical(results['signal'], ['strong', 'weak', 'null'])
results.groupby('method', observed=True).agg(intervals=('covered', 'size'),
                                             coverage=('covered', 'mean'),
                                             mean_length=('length', 'mean')).round(3)
```

The naive intervals undercover, because the variables were chosen with
the same data used for inference. The other three are calibrated. Split
and combination intervals have the same width, $1/\sqrt{1-\pi}$ times
the full-data standard error, but the combination holds no data out:
the parametric perturbation $\omega$ plays the role of the random split.
The carved intervals are much shorter.

Breaking the intervals down by the strength of the signal shows where
the carved intervals gain:

```{code-cell} ipython3
(results.groupby(['signal', 'method'], observed=True)
        .agg(intervals=('covered', 'size'),
             coverage=('covered', 'mean'),
             mean_length=('length', 'mean'))
        .round(3))
```

For the strong signals, the carved intervals are nearly as short as the
naive ones and keep their coverage. These variables would be selected
for almost any $\omega$, so their selection carries little information
and conditioning on it costs little. The naive intervals are fine here
too, but only because selection was a foregone conclusion. For the weak
signals and the nulls that were selected, the naive intervals fail
badly. The carved intervals widen there, by the cost of conditioning on
a selection that might easily have gone the other way, but remain
calibrated and shorter than the split and combination intervals.

The calibration relies on the logistic model. $e$ has the
variance of $y$ under the fitted model, and both the cancellation in
the covariance above and the `scalar_noise` used for carving need that
to match the true variance of the score. Without the model, the
randomization can instead come from a nonparametric bootstrap of the
observations, as in Fry, Panigrahi and Taylor (2025): see
[A misspecified logistic regression](logistic_misspecified.md).

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
