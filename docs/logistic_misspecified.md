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

# A misspecified logistic regression

[Score bootstraps with a linear term](score_bootstrap.md) randomized a
logistic lasso with a parametric perturbation of the score, drawn from
the fitted logistic model, and noted that its calibration relies on the
model being right. Here it is wrong, in both the link and the mean: the
data come from a probit model whose linear predictor has quadratic terms,
and we fit a logistic regression that is linear in the features.

The fitted coefficients then estimate a projection: the logistic
regression closest to the truth, in the sense of the expected
log-likelihood ({ref}`White, 1982 <logmis-refs>`). Inference for the
projection is still possible, but the score no longer varies as the
information says. Its variance is the sandwich.

So we replace the parametric perturbation with a nonparametric one: a
pairs bootstrap of the score at the full-data maximum likelihood
estimate, as in data fission with resampled data
({ref}`Fry, Panigrahi and Taylor, 2025 <logmis-refs>`). Its covariance is
the middle of the sandwich, and it needs no model. We compare the
inference with each perturbation, using the model-based or the sandwich
covariance in `lassoinf`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import norm

from glmnet import GLMNet, LinearTerm
from glmnet.glmnet import GLMNetControl
from glmnet.inference import glmstar_inference_problem, score_perturbation
from glmnet.inference.glm_problem import XTVXOperator
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
```

## Data

The design is that of [Score bootstraps with a linear term](score_bootstrap.md):
$n = 5000$, two strong signals, six weak ones and 22 nulls, and
correlated features with standard deviations from 0.2 to 5. But

$$P(y = 1 \mid x) = \Phi\Big(-0.3 + x^T\beta + \tfrac12\big(u_1^2 - 1\big) + \tfrac12\big(u_2^2 - 1\big)\Big),$$

with $\Phi$ the standard normal distribution function and $u_1, u_2$ the
two strong features in standard units. The link is the probit, not the
logit, and the linear predictor is not linear in $x$.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n, p, rho = 5000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.3, -0.25, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))

def sim_data(n):
    X = (rng.standard_normal((n, p)) @ R) * scale
    u = X[:, :2] / scale[:2]
    eta = -0.3 + X @ beta + 0.5 * (u**2 - 1).sum(1)     # not linear in x
    return X, rng.binomial(1, norm.cdf(eta)).astype(float)

X, y = sim_data(n)
```

## The fit, its score and the sandwich

The loss is the logistic negative log-likelihood divided by $n$, as
`GLMNet` scales it. With $\tilde x_i = (1, x_i)$ and $\hat\mu_i$ the
fitted means of the unpenalized logistic regression on all $p$
features, observation $i$ contributes $s_i = \tilde x_i(y_i - \hat\mu_i)/n$
to the score. The model-based covariance of the score is the information
$Q/n$, with $Q = \sum_i \hat\mu_i(1-\hat\mu_i)\,\tilde x_i\tilde x_i^T/n$
the Hessian of the loss; the sandwich's middle is
$\sum_i s_i s_i^T$.

```{code-cell} ipython3
def logit_fit(X, y):
    """
    The unpenalized logistic regression on X: coefficients (without the intercept),
    model-based and sandwich standard errors, and fitted means.
    """
    X1 = sm.add_constant(X, has_constant='add')
    fit = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
    b, mu = fit.params, fit.fittedvalues
    H_inv = np.linalg.inv(X1.T @ (X1 * (mu * (1 - mu))[:, None]))
    meat = X1.T @ (X1 * ((y - mu)**2)[:, None])
    return (b[1:], np.sqrt(np.diag(H_inv))[1:],
            np.sqrt(np.diag(H_inv @ meat @ H_inv))[1:], mu)
```

On this data set the sandwich standard errors of the two strong
coefficients are larger than the model-based ones:

```{code-cell} ipython3
b_hat, se_model, se_sandwich, mu_hat = logit_fit(X, y)
pd.DataFrame({'model': se_model, 'sandwich': se_sandwich, 'ratio': se_sandwich / se_model},
             index=[f'X{j}' for j in range(p)]).head(8).round(4)
```

The sandwich is right. Over many data sets the score at the projection,
fitted here to a large sample, varies by about the sandwich, not the
information. Binary data cannot be overdispersed, so this comes from
the mean alone: where the logistic fit misses the curvature, the squared
residuals $(y_i - \mu_i)^2$ are not $\mu_i(1 - \mu_i)$ on average.

```{code-cell} ipython3
pop_X, pop_y = sim_data(1000000)
pop_X1 = sm.add_constant(pop_X)
projection = sm.GLM(pop_y, pop_X1, family=sm.families.Binomial()).fit().params

scores, infos = [], []
for _ in range(1000):
    X_s, y_s = sim_data(n)
    X1_s = sm.add_constant(X_s)
    mu_s = 1 / (1 + np.exp(-X1_s @ projection))
    scores.append(X1_s.T @ (y_s - mu_s))
    infos.append(np.sum(X1_s**2 * (mu_s * (1 - mu_s))[:, None], 0))
X1 = sm.add_constant(X)
pd.DataFrame({'Var(score) / information': np.var(scores, 0, ddof=1) / np.mean(infos, 0),
              'sandwich / information (this data set)':
                  np.sum(X1**2 * ((y - mu_hat)**2)[:, None], 0)
                  / np.sum(X1**2 * (mu_hat * (1 - mu_hat))[:, None], 0)},
             index=['intercept'] + [f'X{j}' for j in range(p)]).head(9).round(2)
```

## Two perturbations

Both perturbations are linear terms $-\omega$ for the lasso fit to $y$,
with $\omega$ a draw of the score scaled by $\sqrt{(1-\pi)/\pi}$, as in
[Score bootstraps with a linear term](score_bootstrap.md):

- **parametric**: $\omega = \sqrt{(1-\pi)/\pi}\;\sum_i \tilde x_i\,(y^*_{1i} - y^*_{2i})/(\sqrt 2\, n)$,
  with $y^*_1, y^*_2$ independent Bernoulli($\hat\mu$) replicates, drawn by
  `score_perturbation`. Given $\hat\mu$, its covariance is $(1-\pi)/\pi$
  times the information $Q/n$: it comes from the model.
- **pairs bootstrap**: with $M_1, M_2$ independent multinomial
  $(n; 1/n, \dots, 1/n)$ counts, the weights of two bootstrap samples of
  the observations,

  $$\omega = \sqrt{\frac{1-\pi}{\pi}}\;\sum_i \frac{M_{1i} - M_{2i}}{\sqrt 2}\, s_i.$$

  $\sum_i M_{1i} s_i$ is the score of the first bootstrap sample at the
  full-data estimate. It is already centered, as the full-data score
  $\sum_i s_i$ is 0 at the estimate, and its covariance given the data is
  $\sum_i s_i s_i^T - (\sum_i s_i)(\sum_i s_i)^T/n = \sum_i s_i s_i^T$.
  Differencing two of them and dividing by $\sqrt 2$ keeps that
  covariance and makes the draw symmetric, as the difference of two
  parametric replicates does above. So $\omega$ has $(1-\pi)/\pi$ times
  the middle of the sandwich: it comes from the residuals.

```{code-cell} ipython3
pi = 0.9

def parametric_perturbation(X, mu_hat):
    eta_hat = np.log(mu_hat / (1 - mu_hat))
    return score_perturbation(X, eta_hat, sm.families.Binomial(), (1 - pi) / pi, rng=rng)

def pairs_perturbation(X, y, mu_hat):
    n = X.shape[0]
    M = rng.multinomial(n, np.full(n, 1 / n), size=2)
    r = np.sqrt((1 - pi) / pi) * (M[0] - M[1]) / np.sqrt(2) * (y - mu_hat) / n
    return LinearTerm(coef=-(X.T @ r), intercept=-r.sum())
```

For the pairs bootstrap the matching covariance of the score is the
sandwich's middle $\sum_i s_i s_i^T$. `glmstar_inference_problem` takes it
as `Sigma`, an array or a `LinearOperator`. Here it is
$\tilde X^T D \tilde X$ with $D$ diagonal, an `XTVXOperator`, which costs
$O(np)$ per product and is never formed:

```{code-cell} ipython3
def sandwich(X, y, mu_hat):
    n = X.shape[0]
    return XTVXOperator(sm.add_constant(X, has_constant='add'),
                        aslinearoperator(diags((y - mu_hat)**2 / n**2)))
```

## The inference

The randomized lasso, the combination and the carving are those of
[Score bootstraps with a linear term](score_bootstrap.md). What changes
is the perturbation, the covariance of the score passed to `lassoinf`
and the standard errors of the combination:

- **model-based**: the parametric perturbation, `Sigma` $= Q/n$ (the
  default of `glmstar_inference_problem`), and model-based standard
  errors;
- **robust**: the pairs bootstrap, `Sigma` the sandwich's middle, and
  sandwich standard errors.

In each, the randomization has `scalar_noise` $= (1-\pi)/\pi$ relative
to `Sigma`. Given the data, that is exact: the parametric draws have
$(1-\pi)/\pi$ times the information, the pairs draws $(1-\pi)/\pi$ times
the sandwich's middle at the same $\hat\mu$. What matters is whether
`Sigma` is the variance of the score, and only the sandwich is.

```{code-cell} ipython3
lam = 0.5 * np.sqrt(2 * np.log(p) / (pi * n))
q = norm.ppf(0.95)  # for nominal 90% intervals

tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def logistic_lasso(X, y, linear_term=None, relax=False, control=None):
    return GLMNet(family=sm.families.Binomial(),
                  lambda_values=np.array([lam]),
                  linear_term=linear_term,
                  relax=relax,
                  control=control or GLMNetControl()).fit(X, y)

def randomized_inference(X, y, robust):
    mu_hat = logit_fit(X, y)[3]
    if robust:
        omega, Sigma = pairs_perturbation(X, y, mu_hat), sandwich(X, y, mu_hat)
    else:
        omega, Sigma = parametric_perturbation(X, mu_hat), None
    fit = logistic_lasso(X, y, linear_term=omega, relax=True, control=tight)
    E = np.nonzero(fit.coefs_[0])[0]
    if len(E) == 0:
        return E, None

    # carving
    info = glmstar_inference_problem(fit, X, y, scalar_noise=(1 - pi) / pi, Sigma=Sigma)
    # row 0 is the intercept
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_.loc[E + 1]
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    b_E, se_model, se_sandwich, _ = logit_fit(X[:, E], y)
    T = (b_E - pi * fit.relaxed_coefs_[0][E]) / (1 - pi)
    se = (se_sandwich if robust else se_model) / np.sqrt(1 - pi)
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the logistic regression of the
population on $X_E$, which we approximate by fitting each selected model
to the sample of a million observations drawn above.

```{code-cell} ipython3
targets = {}

def target(E):
    key = tuple(E)
    if key not in targets:
        X1_E = pop_X1[:, np.r_[0, E + 1]]
        targets[key] = sm.GLM(pop_y, X1_E, family=sm.families.Binomial()).fit().params[1:]
    return targets[key]
```

### One data set

```{code-cell} ipython3
tables = {}
for robust in [False, True]:
    E, intervals = randomized_inference(X, y, robust)
    tables['robust' if robust else 'model-based'] = pd.DataFrame(
        {'lower': intervals['carved'][1], 'upper': intervals['carved'][2],
         'target': target(E)}, index=[f'X{j}' for j in E])
pd.concat(tables, axis=1).round(3)
```

## Calibration

We repeat the experiment and compare, each with model-based and with
robust variances:

- **naive**: select with the lasso on the data, then use the logistic
  regression intervals of the selected model on the same data;
- **split**: select on a random 90% of the observations, then form
  intervals on the other 10%;
- **combination** and **carved**: the parametric perturbation with the
  information, or the pairs bootstrap with the sandwich.

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

    def logit_intervals(method, X, y, E):
        b, se_model, se_sandwich, _ = logit_fit(X[:, E], y)
        record(method, 'model-based', E, b - q * se_model, b + q * se_model)
        record(method, 'robust', E, b - q * se_sandwich, b + q * se_sandwich)

    for _ in range(n_rep):
        X, y = sim_data(n)

        E = np.nonzero(logistic_lasso(X, y).coefs_[0])[0]
        if len(E):
            logit_intervals('naive', X, y, E)

        idx = rng.permutation(n)
        sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
        E = np.nonzero(logistic_lasso(X[sel], y[sel]).coefs_[0])[0]
        if len(E):
            logit_intervals('split', X[held_out], y[held_out], E)

        for robust in [False, True]:
            E, intervals = randomized_inference(X, y, robust)
            if len(E):
                for method, (_, lower, upper) in intervals.items():
                    record(method, 'robust' if robust else 'model-based', E, lower, upper)

    return pd.DataFrame(rows)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py logistic_misspecified`
and saved in `docs/results/logistic_misspecified.csv`.

```{code-cell} ipython3
results = load_results('logistic_misspecified')
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

The naive intervals fail as before: they never cover the selected
nulls, and cover the weak signals less than 65% of the time. With
model-based variances they also undercover the strong signals, about
86%, which the sandwich corrects.

For the strong signals the model-based variance is the wrong one. Their
score varies about 1.3 times as much as the information says, so the
model-based standard errors are about $\sqrt{1.3} \approx 1.14$ times too
small, and a nominal 90% interval covers about
$2\Phi(1.645/1.14) - 1 \approx 0.85$ of the time. That is what the
naive, split and carved intervals with model-based variances give for
the strong signals. For the weak signals and the selected nulls, whose
features the curvature leaves alone, the two variances agree and so do
the intervals.

With the pairs bootstrap and the sandwich, the split, combination and
carved intervals cover the strong signals and the selected nulls at
about 90% (Monte Carlo error about 0.007 for the strong signals, 0.025
for the nulls). The combination and carved intervals cover the weak
signals a little less, 0.88 to 0.89 (Monte Carlo error about 0.012), as
they do with the model-based variances; the curvature does not touch
these features, so this is not the variance. The carved intervals are
less than two thirds as long as the split and combination ones for the
weak signals and the nulls, and nearly as short as the naive ones for
the strong signals.

The model-based combination is close to calibrated even for the strong
signals, as in [A misspecified Cox model](cox_misspecified.md). Its
variance is that of $\hat\beta_E$ plus $\pi/(1-\pi) = 9$ times that of
the perturbation, and the parametric perturbation has the variance the
model-based standard errors assume. Only the first part is misjudged,
by a factor of about 1.3, so the whole is misjudged by about
$(1.3 + 9)/(1 + 9) = 1.03$. The carved intervals have no such cushion:
the selection and the full-data score both carry the misspecification.

(logmis-refs)=
## References

- White, H. (1982). Maximum likelihood estimation of misspecified
  models. *Econometrica*, 50(1), 1–25.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates
  under nonstandard conditions. *Proceedings of the Fifth Berkeley
  Symposium on Mathematical Statistics and Probability*, 1, 221–233.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *The Annals of Statistics*, 7(1), 1–26.
- Tian, X., & Taylor, J. (2018). Selective inference with a randomized
  response. *The Annals of Statistics*, 46(2), 679–710.
- Fry, K., Panigrahi, S., & Taylor, J. (2025). Assumption-lean data
  fission with resampled data. *Journal of the American Statistical
  Association*, 120(549). <https://doi.org/10.1080/01621459.2024.2412172>
