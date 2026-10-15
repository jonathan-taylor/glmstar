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
data come from a probit model whose linear predictor curves in the two
strong features, and we fit a logistic regression that is linear in the features.

The fitted coefficients then estimate a projection: the logistic
regression closest to the truth, in the sense of the expected
log-likelihood ({ref}`White, 1982 <logmis-refs>`). Inference for the
projection is still possible, but the score no longer varies as the
information says. Its variance is the sandwich.

So we compare the parametric perturbation, drawn from the misspecified
fit, with a nonparametric one: a pairs bootstrap of the score at the
full-data maximum likelihood estimate, as in data fission with resampled
data ({ref}`Fry, Panigrahi and Taylor, 2025 <logmis-refs>`). Its
covariance is the middle of the sandwich, and it needs no model. Each
perturbation is used with its own covariance of the score: the
information for the parametric draws, the sandwich for the pairs
bootstrap.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as student_t
from sklearn.base import clone

from glmnet import GLMNet, LinearTerm
from glmnet.glmnet import GLMNetControl
from glmnet.base import Design, DiagonalOperator
from glmnet.family import GLMFamilySpec, GLMState
from glmnet.inference import glmstar_inference_problem, score_perturbation
from lassoinf import LassoInference

from experiments import load_results, provenance, reproduce
```

## Data

The design is that of [Score bootstraps with a linear term](score_bootstrap.md):
$n = 5000$, two strong signals, six weak ones and 22 nulls, and
correlated features with standard deviations from 0.2 to 5. But

$$P(y = 1 \mid x) = \Phi\Big(-0.3 + x^T\beta + \big(h(u_1) - m\big) + \big(h(u_2) - m\big)\Big),
\qquad h(u) = \begin{cases} u^2 & |u| \le 1, \\ 2|u| - 1 & |u| > 1, \end{cases}$$

with $\Phi$ the standard normal distribution function, $u_1, u_2$ the
two strong features in standard units and $m = E\,h(u_j)$. The link is
the probit, not the logit, and the linear predictor is not linear in
$x$: $h$ is Huber's function, a square near 0 continued linearly, so the
curvature is a V in each strong feature. The features are multivariate
$t$ on 5 degrees of freedom: a correlated normal vector divided by one
random scale per observation.

Capping the square matters for the well-specified probit at the end,
which puts $h(u_1)$ and $h(u_2)$ in its design. With $u^2$ in their place
those columns are dominated by a few extreme observations, and the lasso
shrinks their coefficients so far, about nine standard errors, that the
one-step linearization of `glmstar_inference_problem` breaks down. With
$h$ the shrinkage is about five standard errors, and the carved intervals
hold.

A wrong link alone does little harm. In simulations not shown here, a
probit or complementary log-log truth with a linear predictor left the
score varying almost exactly as the logistic information says, with
normal or heavy-tailed features: the logistic curve takes up the
difference over most of the range of $x^T\beta$. What breaks the
information is a mean that the fit misses where the features are large,
and the heavy tails put many observations there, with all their features
large at once.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n, p, rho = 5000, 30, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))).T
scale = np.exp(rng.uniform(np.log(0.2), np.log(5), p))   # standard deviations of the features
beta = np.r_[0.3, -0.25, 0.09, -0.07, 0.06, 0.05, -0.04, 0.03, np.zeros(p - 8)] / scale
signal = np.array(['strong'] * 2 + ['weak'] * 6 + ['null'] * (p - 8))
df = 5   # degrees of freedom of the multivariate t features

def h(u):
    """Huber's function: u**2 for |u| <= 1, continued linearly."""
    return np.where(np.abs(u) <= 1, u**2, 2 * np.abs(u) - 1)

m = student_t(df, scale=np.sqrt((df - 2) / df)).expect(h)   # E h(u_j), u_j of variance 1

def sim_data(n):
    Z = rng.standard_normal((n, p)) / np.sqrt(rng.chisquare(df, (n, 1)) / (df - 2))
    X = (Z @ R) * scale
    u = X[:, :2] / scale[:2]
    eta = -0.3 + X @ beta + (h(u) - m).sum(1)     # not linear in x
    return X, rng.binomial(1, norm.cdf(eta)).astype(float)

X, y = sim_data(n)
```

## The fit, its score and the sandwich

The loss is the negative log-likelihood divided by $n$, as `GLMNet`
scales it. With $\tilde x_i = (1, x_i)$, the fitted linear predictor
$\hat\eta_i$ and mean $\hat\mu_i$ of the unpenalized fit on all the
features, $h = d\mu/d\eta$ and $V = \mu(1-\mu)$, observation $i$
contributes $s_i = \hat r_i\,\tilde x_i/n$ to the score, with score
residual $\hat r_i = \hat a_i(y_i - \hat\mu_i)$ and $a = h/V$. For the
logit $a = 1$. The model-based covariance of the score is the
information $Q/n$, with $Q = \sum_i (\hat h_i^2/\hat V_i)\,\tilde x_i\tilde x_i^T/n$;
the sandwich's middle is $\sum_i s_i s_i^T$.

Everything below takes a `GLMNet` estimator, which carries the family:
`logit_model` here and, at the end, `probit_model`. `unpenalized` is its fit at $\lambda = 0$. `score_parts`
evaluates a fit as `GLMNet` does, in a `GLMState`, and takes the score
residuals from its `logl_score` and the information weights
$w\,h^2/V$ from the family's `information`, as for `CoxFamilySpec`.
`Design.add_constant(X)` is $\tilde X = [\mathbf{1}, X]$ as a glmstar
`Design`, like statsmodels' `add_constant` but without copying $X$: it
multiplies the coefficients of a fit on the original scale, and its
`quadratic_form` gives $\tilde X^T G \tilde X$. `glm_summary` turns these
into standard errors.

```{code-cell} ipython3
logit_model = GLMNet(family=sm.families.Binomial())
probit_model = GLMNet(family=sm.families.Binomial(link=sm.families.links.Probit()))
tight = GLMNetControl(epsnr=1e-12, thresh=1e-14)

def unpenalized(model, X, y, linear_term=None, exclude=()):
    """The fit of model at lambda = 0, with a linear term and excluded columns."""
    return clone(model).set_params(lambda_values=np.array([0.]), linear_term=linear_term,
                                   exclude=list(exclude), control=tight).fit(X, y)

def score_parts(fit, X, y):
    """
    At a fit: the design D = [1, X], the score residuals r = a (y - mu) and the
    information weights W (a DiagonalOperator), so that the score is D^T r and the
    information D^T W D.
    """
    family = GLMFamilySpec(base=fit.family)
    D = Design.add_constant(X)
    state = GLMState(coef=fit.coefs_[0], intercept=fit.intercepts_[0])
    state.update(D, family, None)
    return D, state.logl_score(family, y, np.ones(X.shape[0])), family.information(state)

def glm_summary(fit, X, y):
    """
    For an unpenalized fit: coefficients, model-based and sandwich standard errors, the
    linear predictor and the score residuals.
    """
    D, r, W = score_parts(fit, X, y)
    H_inv = np.linalg.inv(D.quadratic_form(W))
    meat = D.quadratic_form(DiagonalOperator(r**2))
    return (fit.coefs_[0], np.sqrt(np.diag(H_inv))[1:],
            np.sqrt(np.diag(H_inv @ meat @ H_inv))[1:],
            D @ np.r_[fit.intercepts_[0], fit.coefs_[0]], r)
```

On this data set the sandwich standard errors of the logistic fit are
larger than the model-based ones, most of all for the two strong
coefficients:

```{code-cell} ipython3
pilot = unpenalized(logit_model, X, y)
b_hat, se_model, se_sandwich, eta_hat, r_hat = glm_summary(pilot, X, y)
pd.DataFrame({'model': se_model, 'sandwich': se_sandwich, 'ratio': se_sandwich / se_model},
             index=[f'X{j}' for j in range(p)]).head(10).round(4)
```

The sandwich is right. Over many data sets the score at the projection,
fitted here to a large sample, varies by about the sandwich, not the
information. Binary data cannot be overdispersed, so this comes from
the mean alone: where the logistic fit misses the truth, the squared
residuals $(y_i - \mu_i)^2$ are not $\mu_i(1 - \mu_i)$ on average.

```{code-cell} ipython3
pop_X, pop_y = sim_data(1000000)
projection = unpenalized(logit_model, pop_X, pop_y)

scores, infos = [], []
for _ in range(1000):
    X_s, y_s = sim_data(n)
    D_s, r_s, W_s = score_parts(projection, X_s, y_s)
    scores.append(D_s.T @ r_s)
    infos.append(np.diag(D_s.quadratic_form(W_s)))
D, _, W_hat = score_parts(pilot, X, y)
pd.DataFrame({'Var(score) / information': np.var(scores, 0, ddof=1) / np.mean(infos, 0),
              'sandwich / information (this data set)':
                  np.diag(D.quadratic_form(DiagonalOperator(r_hat**2)))
                  / np.diag(D.quadratic_form(W_hat))},
             index=['intercept'] + [f'X{j}' for j in range(p)]).head(11).round(2)
```

## Two perturbations

Both perturbations are linear terms $-\omega$ for the lasso fit to $y$,
with $\omega$ a draw of the score scaled by $\sqrt{(1-\pi)/\pi}$, as in
[Score bootstraps with a linear term](score_bootstrap.md):

- **parametric**: $\omega = \sqrt{(1-\pi)/\pi}\;\sum_i \hat a_i\,\tilde x_i\,(y^*_{1i} - y^*_{2i})/(\sqrt 2\, n)$,
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

def parametric_perturbation(pilot, X, eta_hat):
    return score_perturbation(X, eta_hat, pilot.family, (1 - pi) / pi, rng=rng)

def pairs_perturbation(X, r_hat):
    n = X.shape[0]
    M = rng.multinomial(n, np.full(n, 1 / n), size=2)
    e = np.sqrt((1 - pi) / pi) * (M[0] - M[1]) / np.sqrt(2) * r_hat / n
    omega = Design.add_constant(X).T @ e          # [intercept, coefficients]
    return LinearTerm(coef=-omega[1:], intercept=-omega[0])
```

For the pairs bootstrap the matching covariance of the score is the
sandwich's middle $\sum_i s_i s_i^T$. `glmstar_inference_problem` takes it
as `Sigma`, an array or a `LinearOperator`. Here it is the product
$\tilde X^T \mathrm{diag}(\hat r^2/n^2)\, \tilde X$ of the `Design` and a
`DiagonalOperator`, a `LinearOperator` that costs $O(np)$ per product
and is never formed:

```{code-cell} ipython3
def sandwich(X, r_hat):
    n = X.shape[0]
    D = Design.add_constant(X)
    return D.T @ DiagonalOperator(r_hat**2 / n**2) @ D
```

## The inference

The randomized lasso, the combination and the carving are those of
[Score bootstraps with a linear term](score_bootstrap.md), for either
link. There are two versions:

- **parametric**: the parametric perturbation, `Sigma` $= Q/n$ (the
  default of `glmstar_inference_problem`), and model-based standard
  errors for the combination;
- **pairs**: the pairs bootstrap, `Sigma` the sandwich's middle, and
  sandwich standard errors.

In each, the randomization has `scalar_noise` $= (1-\pi)/\pi$ relative
to `Sigma`. Given the data, that is exact: the parametric draws have
$(1-\pi)/\pi$ times the information, the pairs draws $(1-\pi)/\pi$ times
the sandwich's middle at the same $\hat\mu$. What matters is whether
`Sigma` is the variance of the score.

As in [Probit regression](probit_regression.md), the selected model $E$
is the one `lassoinf` conditions on, and $\hat\beta^*_E$ is the
unpenalized fit on $X_E$ with the linear term $-\omega$.

```{code-cell} ipython3
q = norm.ppf(0.95)  # for nominal 90% intervals

def lasso(model, X, y, linear_term=None, control=None):
    lam = 0.5 * np.sqrt(2 * np.log(X.shape[1]) / (pi * X.shape[0]))
    return clone(model).set_params(lambda_values=np.array([lam]), linear_term=linear_term,
                                   control=control or GLMNetControl()).fit(X, y)

def randomized_inference(model, X, y, pairs):
    pilot = unpenalized(model, X, y)
    _, _, _, eta_hat, r_hat = glm_summary(pilot, X, y)
    if pairs:
        omega, Sigma = pairs_perturbation(X, r_hat), sandwich(X, r_hat)
    else:
        omega, Sigma = parametric_perturbation(pilot, X, eta_hat), None
    fit = lasso(model, X, y, linear_term=omega, control=tight)
    if not np.any(fit.coefs_[0]):
        return [], None

    # carving
    info = glmstar_inference_problem(fit, X, y, scalar_noise=(1 - pi) / pi, Sigma=Sigma)
    summary = LassoInference(**info.lasso_args(), level=0.9).summary_
    summary = summary.loc[summary.index > 0]       # row 0 is the intercept
    E = summary.index.values - 1
    if len(E) == 0:
        return E, None
    carved = (summary['beta_hat'].values, summary['lower_conf'].values,
              summary['upper_conf'].values)

    # combination
    b_E, se_model, se_sandwich, _, _ = glm_summary(unpenalized(model, X[:, E], y), X[:, E], y)
    others = sorted(set(range(X.shape[1])) - set(E))
    b_star = unpenalized(model, X, y, linear_term=omega, exclude=others).coefs_[0][E]
    T = (b_E - pi * b_star) / (1 - pi)
    se = (se_sandwich if pairs else se_model) / np.sqrt(1 - pi)
    combination = (T, T - q * se, T + q * se)
    return E, {'combination': combination, 'carved': carved}
```

The targets $\beta_E$ are projections: the regression of the population
on $X_E$, with the link of the model, which we approximate by fitting
each selected model to the sample of a million observations drawn
above. `pop_design` holds the columns of every design used below.

```{code-cell} ipython3
def curvature_design(X):
    """X with h of the two strong features, for the probit below."""
    return np.column_stack([X, h(X[:, :2] / scale[:2])])

pop_design = curvature_design(pop_X)
del pop_X
targets = {}

def target(model, E):
    key = (type(model.family.link).__name__, tuple(E))
    if key not in targets:
        targets[key] = unpenalized(model, pop_design[:, E], pop_y).coefs_[0]
    return targets[key]
```

### One data set

```{code-cell} ipython3
tables = {}
for pairs in [False, True]:
    E, intervals = randomized_inference(logit_model, X, y, pairs)
    tables['pairs' if pairs else 'parametric'] = pd.DataFrame(
        {'lower': intervals['carved'][1], 'upper': intervals['carved'][2],
         'target': target(logit_model, E)}, index=[f'X{j}' for j in E])
pd.concat(tables, axis=1).round(3)
```

## Calibration

We repeat the experiment and compare:

- **naive**: select with the lasso on the data, then use the intervals
  of the selected model on the same data, with model-based or sandwich
  standard errors;
- **split**: select on a random 90% of the observations, then form
  intervals on the other 10%, with model-based or sandwich standard
  errors;
- **combination** and **carved**: the parametric version or the pairs
  version above.

For every selected coefficient we record whether its nominal 90%
interval covers $\beta_{E,j}$, and the interval's length per standard
deviation of the feature. `run_calibration` does this for a model and a
design; `calibration` runs it for the misspecified logistic fit.

```{code-cell} ipython3
def run_calibration(n_rep, model, design, labels, sd):
    rows = []

    def record(method, variance, E, lower, upper):
        for j, lo, hi, t in zip(E, lower, upper, target(model, E)):
            rows.append({'method': method, 'variance': variance, 'signal': labels[j],
                         'covered': lo <= t <= hi, 'length': (hi - lo) * sd[j]})

    def glm_intervals(method, X, y, E):
        b, se_model, se_sandwich, _, _ = glm_summary(unpenalized(model, X[:, E], y), X[:, E], y)
        record(method, 'model-based', E, b - q * se_model, b + q * se_model)
        record(method, 'sandwich', E, b - q * se_sandwich, b + q * se_sandwich)

    for _ in range(n_rep):
        X, y = sim_data(n)
        X = design(X)

        E = np.nonzero(lasso(model, X, y).coefs_[0])[0]
        if len(E):
            glm_intervals('naive', X, y, E)

        idx = rng.permutation(n)
        sel, held_out = idx[:int(pi * n)], idx[int(pi * n):]
        E = np.nonzero(lasso(model, X[sel], y[sel]).coefs_[0])[0]
        if len(E):
            glm_intervals('split', X[held_out], y[held_out], E)

        for pairs in [False, True]:
            E, intervals = randomized_inference(model, X, y, pairs)
            if len(E):
                for method, (_, lower, upper) in intervals.items():
                    record(method, 'sandwich' if pairs else 'model-based', E, lower, upper)

    return pd.DataFrame(rows)

def calibration(n_rep=300):
    return run_calibration(n_rep, logit_model, lambda X: X, signal, scale)
```

The experiment is run locally, not when the docs are built: the results
below were computed with `python docs/experiments.py logistic_misspecified`
and saved in `docs/results/logistic_misspecified.csv`. For the
combination and carved intervals, `model-based` is the parametric
version and `sandwich` the pairs version.

```{code-cell} ipython3
def summarize(results):
    """Coverage, and the number of intervals with their mean length."""
    results['method'] = pd.Categorical(results['method'],
                                       ['naive', 'split', 'combination', 'carved'])
    results['variance'] = pd.Categorical(results['variance'], ['model-based', 'sandwich'])
    results['signal'] = pd.Categorical(results['signal'], ['curvature', 'strong', 'weak', 'null'])
    groups = results.groupby(['method', 'variance', 'signal'], observed=True)
    return (groups['covered'].mean().unstack('signal').round(3),
            pd.concat({'intervals': groups.size().unstack('signal'),
                       'mean length': groups['length'].mean().unstack('signal').round(3)},
                      axis=1))

results = load_results('logistic_misspecified')
display(provenance(results))
display(reproduce(results))
coverage, lengths = summarize(results)
coverage
```

The number of intervals, and their mean length per standard deviation of
the feature:

```{code-cell} ipython3
lengths
```

RESULTS-DISCUSSION

## A well-specified probit

Fitting the right model to the same data shows the two perturbations
when the parametric one is right. The truth is a probit model whose
linear predictor is linear in the features and in $h(u_1)$, $h(u_2)$,

$$-0.3 - 2m + x^T\beta + h(u_1) + h(u_2),$$

so we add the two columns $h(u_1)$ and $h(u_2)$ to the design and fit a
probit lasso to all 32 columns, with $\lambda$ chosen by the same rule and no
special penalty factors. The parameters are not those above: the targets are probit
coefficients, the projections of the probit regression on $X_E$, and
they are the true coefficients when $E$ contains every signal. The two
columns of $h$ are labelled `curvature`.

On this data set the sandwich and model-based standard errors of the
probit fit agree:

```{code-cell} ipython3
W = curvature_design(X)
signal_W = np.r_[signal, ['curvature'] * 2]
scale_W = np.r_[scale, pop_design[:, p:].std(0)]
_, se_model_W, se_sandwich_W, _, _ = glm_summary(unpenalized(probit_model, W, y), W, y)
pd.DataFrame({'model': se_model_W, 'sandwich': se_sandwich_W,
              'ratio': se_sandwich_W / se_model_W},
             index=[f'X{j}' for j in range(p)] + ['h(u1)', 'h(u2)']).iloc[np.r_[0:8, p:p + 2]].round(4)
```

```{code-cell} ipython3
def calibration_well_specified(n_rep=300):
    return run_calibration(n_rep, probit_model, curvature_design, signal_W, scale_W)
```

The results below were computed with
`python docs/experiments.py logistic_misspecified --function calibration_well_specified`
and saved in `docs/results/logistic_misspecified-calibration_well_specified.csv`.

```{code-cell} ipython3
results_W = load_results('logistic_misspecified', 'calibration_well_specified')
display(provenance(results_W))
display(reproduce(results_W))
coverage_W, lengths_W = summarize(results_W)
coverage_W
```

The number of intervals, and their mean length per standard deviation of
the feature:

```{code-cell} ipython3
lengths_W
```

WELL-SPECIFIED-DISCUSSION

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
