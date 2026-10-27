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

# Generalized additive models

A *generalized additive model* (GAM) replaces the linear predictor of a
GLM by a sum of smooth functions of the predictors,

$$g\big(E[y \mid x]\big) = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_p(x_p),$$

each $f_j$ a smoothing spline, a loess curve, a linear function, a categorical's
effects, or a random effect. In a `GAM` the smoothing of each term is
**fixed**: given by its degrees of freedom `df` or its smoothing parameter
`lam`, as R's `gam::gam`. To estimate the smoothing parameters (by REML or
ML, as mgcv), fit a `GLMM` (the page [Generalized linear mixed
models](glmm.md)): it has the same terms, its penalized terms random
effects whose variances it estimates.

`glmnet.gam` fits GAMs in two ways that give the same fit:

- **local scoring with backfitting** (Hastie and Tibshirani, 1986, 1990),
  as R's `gam::gam`: IRLS, each working problem solved by smoothing the
  partial residuals of each term in turn;
- **penalized regression** on the terms' bases (Wood, 2017), as R's
  `mgcv::gam`: each working problem one penalized least squares fit. This
  gives the Bayesian covariance of the fit.

It is built on glmstar's IRLS, so the families are those of `GLM`
(statsmodels'), with weights and offsets, and the Cox model. Each term is
a *smoother* (`glmnet.gam.Smoother`), and a new kind of term is a small
class: see [Smoothers](#smoothers) and [Writing a
smoother](#writing-a-smoother) at the end of the page.

This page follows the GAM part of the ISLR lab on nonlinear modeling
(chapter 7), on its `Wage` data from the `ISLP` package.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data

from glmnet.gam import GAM, GAMControl, s, lo, random_effect as re, linear, categorical, tensor_spline, varying

Wage = load_data('Wage')
Wage[['wage', 'year', 'age', 'education']].head()
```

## A GAM by local scoring

The model of the lab: `wage` as a smooth function of `year` with 4 degrees
of freedom, of `age` with 5, and an effect for each level of `education`.
Each term names a column of the data frame.

```{code-cell} ipython3
gam = GAM(terms=[s('year', df=4), s('age', df=5), categorical('education')])
gam.fit(Wage, Wage['wage'])
gam.summary()
```

In a notebook a `GAM` shows the model it specifies: the response's
distribution and link, the linear predictor, and what each term is
(`model_latex()` gives the LaTeX):

```{code-cell} ipython3
gam
```

`edf` is each term's degrees of freedom, the trace of its smoother at the
final weights; for a spline it is the `df` it was given (the constant is
the intercept's). `lam` is the smoothing parameter that gives it. With
`edf_influence`, mgcv's degrees of freedom of the penalized fit, it is close
to `edf`.

`GAM.plot` draws each term's fit, centered, with dashed bands of 2
pointwise standard errors and a rug of the data, as R's `plot.Gam`:

```{code-cell} ipython3
axes = gam.plot()
axes[0].figure.tight_layout()
```

Wages rise with `year`, and with `age` until about 40, and fall after 60;
holding the others fixed, each level of education adds to the wage.

The fitted terms are `fits_`, one column per term, and the intercept is
`intercept_`: their sum is the linear predictor.

```{code-cell} ipython3
np.allclose(gam.intercept_ + gam.fits_.sum(axis=1), gam.predict(Wage))
```

## Parametric terms: columns, contrasts, interactions

The items of `terms` other than smoothers are the model's parametric
part, as in a formula: a column, an interaction (a tuple of columns), or
any term of modelmatrix's `ModelSpec`, such as `contrast()` and
`poly()`. They make one `ModelSpec`, whose terms enter unpenalized as
blocks of columns. This is mgcv's `wage ~ s(age) + year + education +
poly(year, 2):education`, with no ModelSpec written:

```{code-cell} ipython3
from modelmatrix import contrast, poly, wildcard

gam_p = GAM(terms=[s('age', df=5), 'year', 'education', (poly('year', 2), 'education')])
gam_p.fit(Wage, Wage['wage'])
gam_p.summary()
```

Each term is a row of the summary, in the order given. Its
coefficients, with their standard errors, are named by the ModelSpec's
columns:

```{code-cell} ipython3
gam_p.fixed_effects_.head(8)
```

The contrasts, the interactions and the bases (`poly`, `ns`, `bs`, `pca`)
are modelmatrix's, as for a GLM. A level not seen at `fit` is an error at
`predict`. `wildcard(exclude=[...])` gives the other columns, as R's
`y ~ s(age) + .`. A column that is both a smooth's and a parametric
term's, e.g. `s('age')` and `'age'`, is an error: their linear parts are
not identified. A `ModelSpec` item works too, e.g. for its
`categorical_encoders`.

`linear('x')` and `categorical('f')` remain. `categorical()` backfits a
factor of many levels (more than 50) by its group means, with no design
matrix.

## The smoothing of a term

A spline's smoothing is given either by its degrees of freedom `df`,
matched at each working problem's weights as R's `gam`, or by its
smoothing parameter `lam`, the weight of its roughness penalty
$\frac{\lambda}{2}\int f''(x)^2\,dx$ in the objective (the deviance over
twice the sum of the weights). With `lam` the GAM minimizes that objective.

```{code-cell} ipython3
for lam in [1e-4, 1e-2, 1]:
    fit = GAM(terms=[s('year', df=4), s('age', lam=lam), categorical('education')]).fit(Wage, Wage['wage'])
    print(f'lam = {lam:g}: the spline in age has {fit.edf_.iloc[1]:.2f} degrees of freedom')
```

`lo()` is a loess term (Cleveland, Grosse and Shyu, 1992), with its span
and degree:

```{code-cell} ipython3
gam_lo = GAM(terms=[s('year', df=4), lo('age', span=0.7), categorical('education')])
gam_lo.fit(Wage, Wage['wage'])
gam_lo.summary()
```

Loess has no basis, so this fit has no standard errors: its plot has no
bands.

```{code-cell} ipython3
axes = gam_lo.plot(terms=['lo(age, span=0.7)'])
```

### Backfitting and the joint solve

By default each working problem is solved by *modified backfitting* (Buja,
Hastie and Tibshirani, 1989): the linear parts of all the terms are fitted
together, then the smooths' nonlinear parts are backfitted.
`GAMControl(solver='joint')` instead solves it as one penalized least
squares fit on the terms' bases, side by side, as mgcv. For terms with a
basis (all but loess) the two give the same fit:

```{code-cell} ipython3
joint = GAM(terms=[s('year', df=4), s('age', df=5), categorical('education')],
            control=GAMControl(solver='joint')).fit(Wage, Wage['wage'])
np.abs(joint.fits_ - gam.fits_).max().max()
```

## Standard errors and predictions

When every term has a basis, the fit has mgcv's Bayesian covariance of the
coefficients of the terms' bases, `posterior_covariance_` (Wahba, 1983;
Nychka, 1988; their intervals' coverage is discussed by Marra and Wood,
2012). It gives standard errors of predictions and of the terms' curves:

```{code-cell} ipython3
new = pd.DataFrame({'year': [2005, 2005, 2005],
                    'age': [25, 45, 65],
                    'education': ['4. College Grad'] * 3})
fit, std = gam.predict(new, return_std=True)
pd.DataFrame({'age': new['age'], 'wage': fit, 'std err': std})
```

The partial effect of a term at new values, with its standard errors, is
`partial_effect`; this is what `plot` draws.

```{code-cell} ipython3
ages = np.linspace(20, 70, 6)
curve, std = gam.partial_effect('s(age, df=5)', ages, return_std=True)
pd.DataFrame({'age': ages, 's(age)': curve, 'std err': std})
```

### Sandwich (robust) standard errors

The posterior covariance trusts the model's variance function. A sandwich
covariance (Huber, 1967; White, 1980) instead takes the variance of the
score from the data: `sandwich_covariance()` is $A^{-1} M A^{-1}$, with

- the **bread** $A = X^\top W X + S_\lambda$, the Hessian of the penalized
  objective $\text{deviance} / (2 \sum_i w_i) + \beta^\top S_\lambda \beta / 2$.
  $X$ is `design_.X`: a column of ones, then each term's basis with its
  sum-to-zero constraint absorbed (or centered). $W$ is
  `working_weights_`, the final IRLS weights $v_i \mu'(\eta_i)^2 /
  V(\mu_i)$, with the prior weights normalized to $v_i = w_i / \sum_j
  w_j$. $S_\lambda$ is `penalty_matrix_`, at the fitted smoothing
  parameters. So `posterior_covariance_` is $\phi A^{-1} / \sum_i w_i$;
- the **meat** $M$, the covariance of the score $\sum_i s_i$ of the same
  objective. Its rows $s_i = v_i (y_i - \mu_i) \mu'(\eta_i) / V(\mu_i)
  \, x_i$ are `score_contributions()`, and they sum to $S_\lambda \beta$ at
  the fit. The model-based meat $\phi X^\top W X / \sum_i w_i$ gives
  `frequentist_covariance_` (mgcv's `Ve`). The default is HC0's,
  $\sum_i s_i s_i^\top$. With `groups`, it is the cluster-robust meat
  $\sum_g \big(\sum_{i \in g} s_i\big)\big(\sum_{i \in g} s_i\big)^\top$, with no
  small-sample correction. Or pass a meat of your own, on the scale of
  `score_contributions()`.

The dispersion $\phi$ does not enter the sandwich. `predict`,
`partial_effect` and `plot` take `covariance=`: `'posterior'` (the
default), `'frequentist'`, `'sandwich'` (HC0) or a matrix.

For grouped rows the score contributions of a group are correlated. Here
40 groups share a shock that the GAM leaves out, so the model's errors
are too small, and the clustered ones are not:

```{code-cell} ipython3
rng = np.random.default_rng(1)
n_groups, size = 40, 25
g = np.repeat(np.arange(n_groups), size)
x = rng.uniform(0, 1, n_groups * size)
shock = rng.normal(0, 0.5, n_groups)
grouped = pd.DataFrame({'x': x, 'g': g,
                        'y': np.sin(6 * x) + shock[g] + 0.5 * rng.standard_normal(len(x))})
fit = GAM(terms=[s('x', df=5)]).fit(grouped, grouped['y'])
V_cluster = fit.sandwich_covariance(groups=grouped['g'])
new = pd.DataFrame({'x': [0.1, 0.5, 0.9]})
table = {'x': new['x']}
for name, cov in [('posterior', 'posterior'), ('HC0', 'sandwich'), ('cluster (g)', V_cluster)]:
    table[name] = fit.predict(new, return_std=True, covariance=cov)[1]
```

A meat of your own can come from a bootstrap. For grouped data, resample
the groups' sums of the score contributions (a grouped, or cluster,
bootstrap). Here the meat is the covariance of their resampled totals:

```{code-cell} ipython3
sums = pd.DataFrame(fit.score_contributions()).groupby(grouped['g'].values).sum().values
totals = np.array([sums[rng.integers(0, n_groups, n_groups)].sum(0) for _ in range(1000)])
V_boot = fit.sandwich_covariance(meat=np.cov(totals, rowvar=False))
table['grouped bootstrap'] = fit.predict(new, return_std=True, covariance=V_boot)[1]
pd.DataFrame(table)
```

Which to use:

- independent rows whose variance the family may get wrong: HC0;
- grouped rows (repeated measures, clusters): `groups=`, or a grouped
  bootstrap with few or unequal groups;
- serial correlation: a meat from sums over blocks of consecutive rows (a
  block bootstrap);
- a `GLMM` whose random effects already model the grouping: the posterior
  covariance, the default.

With penalties, the sandwich is the variance around the penalized fit. It
says nothing about the penalty's bias, which the posterior covariance
partly accounts for. mgcv's `vcov(m, sandwich=TRUE, freq=TRUE)` is HC0's
sandwich times $n / (n - \text{edf})$ (with unit prior weights). Without
`freq=TRUE` it adds `Vp - Ve`, here `posterior_covariance_ -
frequentist_covariance_`.

## A logistic GAM

The lab also models the probability that a worker earns more than 250
(thousand dollars a year). No worker without a high school diploma does, so
the effect of that level would be infinite: as in the lab, those workers
are left out.

```{code-cell} ipython3
high = Wage[Wage['education'] != '1. < HS Grad'].copy()
high['high'] = (high['wage'] > 250).astype(float)
gam_b = GAM(terms=[s('year', df=4), s('age', df=5), categorical('education')],
            family=sm.families.Binomial())
gam_b.fit(high, high['high'])
axes = gam_b.plot()
axes[0].figure.tight_layout()
```

The curves are on the scale of the linear predictor, the log odds. The
probabilities, with standard errors by the delta method:

```{code-cell} ipython3
new = pd.DataFrame({'year': 2008, 'age': [30, 45, 60], 'education': '5. Advanced Degree'})
prob, std = gam_b.predict(new, return_std=True)
pd.DataFrame({'age': new['age'], 'P(wage > 250)': prob, 'std err': std})
```

Another link, or a family of one's own, is a `glmnet.gam.JaxFamily`: its
inverse link and unit deviance in JAX, for fixed smoothing as for REML
([GLMMs](glmm.md)).

## Smooths of several columns: tensor products

`tensor_spline('age', 'year')` is a smooth function of both columns, the
tensor product of a cubic regression spline in each (mgcv's
`te(age, year, bs="cr")`), with a penalty per column. Each column then has
its own smoothing parameter (`lam`, one per column), and the term does not
depend on the columns' scales. In place of the two smooths, it models an
interaction of age and year:

```{code-cell} ipython3
gam_te = GAM(terms=[tensor_spline('age', 'year', k=(6, 4), lam=[1., 1.]),
                    categorical('education')])
gam_te.fit(Wage, Wage['wage'])
gam_te.summary()
```

A two-column term is drawn as filled contours:

```{code-cell} ipython3
ax = gam_te.plot(terms=[0])[0]
ax.figure.tight_layout()
```

Its penalties' null space is spanned by $1$, age, year and their product:
however large its smoothing parameters, the term keeps a linear
age-by-year interaction. (A GLMM estimates the two smoothing parameters:
[Generalized linear mixed models](glmm.md).)

## Varying coefficients

`varying(term, by=...)` lets a smooth term vary with another column, the
varying-coefficient models of Hastie and Tibshirani (1993); it is mgcv's
`by=`. With a categorical `by`, it fits a curve for each level:
`varying(s('age'), by='jobclass')` is mgcv's `s(age, by=jobclass)`. Each
curve is zero off its level; as mgcv's, each is constrained like the term
alone (it sums to zero over the data's ages), so the model includes
`categorical('jobclass')` for the levels' means, as mgcv's
`wage ~ jobclass + s(age, by=jobclass)`. Each level has its own smoothing
parameter, here given (one for both):

```{code-cell} ipython3
by_class = [categorical('jobclass'), varying(s('age'), by='jobclass', lam=2.),
            s('year', df=4), categorical('education')]
gam_by = GAM(terms=by_class).fit(Wage, Wage['wage'])
gam_by.terms_[1].level_edf()
```

The term is drawn as a curve per level:

```{code-cell} ipython3
ax = gam_by.plot(terms=[1])[0]
ax.figure.tight_layout()
```

`by` is categorical, whatever its values (one coded 0 and 1 too). With
`as_categorical=False` it is a number and the term is `z f(x)`: the
coefficient of `z` is a smooth function of `x` (mgcv's `s(x, by=z)`, not
centered). Any smooth can vary: `varying(cyclic_spline('hour', period=(0,
24)), by='workingday')`, `varying(tensor_spline('lon', 'lat'), by='year')`,
`varying(re('school'), by='sector')`, `varying(lo('x'), by='f')` (a loess
per level, by backfitting only). mgcv's `s(age, by=jobclass, bs="cr")` is
`varying(cubic_regression_spline('age'), by='jobclass')`.

## Random effects

A random effect is a term like the smooths: `random_effect(group)`
(imported here as `re`, `from glmnet.gam import random_effect as re`) gives each group
its own intercept, penalized by `lam / 2` times their sum of squares (a
ridge penalty, the Gaussian prior of a mixed model), and `re(group, by=x)`
its own slope in `x`. With a list of columns, `re(['batch', 'cask'])` gives
each combination its own effect, so nested effects, lme4's `(1 |
batch/cask)`, are `re('batch') + re(['batch', 'cask'])`. In a GAM the
penalty is given; its variance, estimated, is a GLMM's ([Generalized
linear mixed models](glmm.md)).

Here marital status is a random effect, its levels' effects shrunk towards
0 the more the smaller the group:

```{code-cell} ipython3
gam_re = GAM(terms=[s('year', df=4), s('age', df=5), categorical('education'),
                    re('maritl', lam=0.01)])
gam_re.fit(Wage, Wage['wage'])
pd.DataFrame({'workers': Wage['maritl'].value_counts().sort_index(),
              'effect': gam_re.terms_[3].effects_})
```

## A Cox model

With glmstar's `CoxFamily`, the GAM is a proportional hazards model with
smooth terms, fitted by maximizing the penalized partial likelihood (with
Efron's or Breslow's ties, start times and strata). The response is a data
frame with the times and the status. On the `BrainCancer` data, the hazard
of death as a smooth function of the gross tumor volume `gtv`, with 3
degrees of freedom, the Karnofsky index `ki` linear and an effect for the
diagnosis:

```{code-cell} ipython3
from glmnet.cox import CoxFamily

Brain = load_data('BrainCancer').dropna()
cox = GAM(terms=[s('gtv', df=3), linear('ki'), categorical('diagnosis')],
          family=CoxFamily(event_id='time', status_id='status', tie_breaking='efron'))
cox.fit(Brain, Brain[['time', 'status']])
cox.summary()
```

```{code-cell} ipython3
axes = cox.plot()
axes[0].figure.tight_layout()
```

The curves are on the scale of the log hazard; `predict` gives the linear
predictor (`'link'`) or the relative risk (`'response'`). (A GLMM, by
REML as mgcv's `cox.ph`, finds the tumor volume's effect to be a line.)

## Choosing a smoother by cross-validation

A `GAM` is a scikit-learn estimator, and so are its terms: a term is
`terms__<j>` among its parameters, and the term's arguments are
`terms__<j>__<arg>`, as a `Pipeline`'s steps. `GridSearchCV` can choose the
smoother of a column, or its degrees of freedom, by the held-out fit,
`score` (minus half the deviance on the held-out rows; for Cox, their
partial log-likelihood). Here, the degrees of freedom of the spline in age:

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV, KFold

folds = KFold(5, shuffle=True, random_state=0)
search = GridSearchCV(gam, {'terms__1__df': [2, 3, 5, 8, 12]}, cv=folds)
search.fit(Wage, Wage['wage'])
pd.Series(search.cv_results_['mean_test_score'],
          index=search.cv_results_['param_terms__1__df'], name='CV score')
```

and the smoother of age itself (the two of [Writing a
smoother](#writing-a-smoother), below, are candidates too):

```{code-cell} ipython3
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from glmnet.gam import TransformerSmoother, RegressorSmoother

candidates = [s('age', df=5), lo('age', span=0.5),
              TransformerSmoother('age', SplineTransformer(n_knots=20), lam=0.1),
              RegressorSmoother('age', DecisionTreeRegressor(max_depth=3))]
search = GridSearchCV(gam, {'terms__1': candidates}, cv=folds).fit(Wage, Wage['wage'])
pd.Series(search.cv_results_['mean_test_score'],
          index=[t.name for t in candidates], name='CV score')
```

The three smooths score within a few dozen units of each other (a sum
over 3,000 workers), and the tree, a step function, is clearly worse.

Several terms are tuned together by a grid over each: here the smoother of
`year` (linear, or a spline of 2 or 4 degrees of freedom) and that of `age`
(splines, loess and a P-spline), 18 models:

```{code-cell} ipython3
grid = {'terms__0': [linear('year'), s('year', df=2), s('year', df=4)],
        'terms__1': [s('age', df=3), s('age', df=5), s('age', df=8),
                     lo('age', span=0.3), lo('age', span=0.6),
                     TransformerSmoother('age', SplineTransformer(n_knots=20), lam=0.1)]}
search = GridSearchCV(gam, grid, cv=folds).fit(Wage, Wage['wage'])
results = pd.DataFrame({'year': [t.name for t in search.cv_results_['param_terms__0']],
                        'age': [t.name for t in search.cv_results_['param_terms__1']],
                        'CV score': search.cv_results_['mean_test_score'],
                        'rank': search.cv_results_['rank_test_score']})
results.sort_values('rank').head(8)
```

`search.best_estimator_` is the GAM refitted on all the rows with the
chosen smoothers; `year` enters linearly:

```{code-cell} ipython3
[term.name for term in search.best_estimator_.terms]
```

The folds should suit the model. A random effect predicts 0 for a group
not in the training folds, so `GroupKFold` (by the group) scores the
prediction of new groups, and `KFold` that of groups seen; rows
correlated in time need blocked folds (`TimeSeriesSplit`); and with
`weight_id` or `offset_id` the response's DataFrame carries them through
the folds.

(smoothers)=
## Smoothers

Each term of a GAM is a `glmnet.gam.Smoother`: an object that, given the
weights of a working problem, fits a function of its column(s) to partial
residuals. Backfitting needs nothing else. Its methods, in the order a fit
calls them:

| method | what it does |
|---|---|
| `setup(x)` | set up on the training data's column(s) `x` (knots, levels, ...) |
| `set_weights(w)` | the weights of the next working problem (`self.w_`) |
| `smooth(r)` | the weighted fit to the partial residuals `r`, at the training rows |
| `edf()` | its degrees of freedom at the current weights, without the constant |
| `finalize(f, r, coef)` | keep the final fit `f` (and coefficients), to predict |
| `predict(x)` | the fitted function at new values `x` |

and `name` (from `_label`, `'s'` for `s('age')`), `columns` (the columns
it uses), `centered` (whether backfitting centers its fits, the intercept
taking the mean; a term identified by its penalty, a random effect, is not)
and `penalty(f, coef)`, its part of the objective (0 by default).

Most smoothers are also **penalized smoothers** (`PenalizedSmoother`):
the fit is $B\hat c$, with $B$ the smoother's *basis* (`basis()`, the
columns of the functions it fits) and

$$\hat c = \arg\min_c \tfrac12 \sum_i w_i (r_i - (Bc)_i)^2 +
\sum_m \tfrac{\lambda_m}{2} c^\top S_m c,$$

$S_m$ its *penalties* (`penalty_matrices()`), $\lambda_m$ its smoothing
parameters (`lam`). The basis and the penalties are all that the joint
solve (`GAMControl(solver='joint')`), the posterior covariance (standard
errors, `predict(..., return_std=True)`) and a GLMM need. A smoother
without them (loess) is fitted by backfitting only, without standard
errors.

| term | smoother | basis | penalties |
|---|---|---|---|
| `s(x)` | `SplineSmoother` | natural cubic spline | $\int f''^2$ |
| `cyclic_spline(x)` | `CyclicSplineSmoother` | cyclic cubic spline | $\int f''^2$ |
| `tensor_spline(x, z)` | `TensorSplineSmoother` | tensor product | one per column |
| `gaussian_process(x, ...)` | `GaussianProcessSmoother` | low rank kriging | the kernel's |
| `random_effect(g)`, `random_effect(g, by=x)` | `RandomEffectSmoother` | group indicators (times `x`) | the identity |
| `correlated_random_effect(g, ...)` | `CorrelatedRandomEffectSmoother` | indicators and slopes | a precision |
| `penalized_columns(cols, S)` | `PenalizedColumnsSmoother` | the columns | given |
| `varying(term, by=...)` | `VaryingSmoother` | the term's, per level | the term's, per level |
| `linear(x)`, `categorical(g)` | `LinearSmoother`, `CategoricalSmoother` | $x$, the indicators | none |
| `lo(x)` | `LoessSmoother` | none | none |
| a scikit-learn transformer | `TransformerSmoother` | its features | given, difference, ridge |
| a scikit-learn regressor | `RegressorSmoother` | none | none |

(The former names, `SplineTerm`, `RandomEffectTerm`, ..., still work, with
a FutureWarning.)

(writing-a-smoother)=
## Writing a smoother

A new kind of term is a subclass of one of the two. A `Smoother` gives
the smoothing itself; a `PenalizedSmoother` gives only a basis and
penalties, and inherits the rest.

### A smoother: means in bins

A step function, the weighted mean of the residuals in each of `n_bins`
bins of the column (quantiles of the training data). It has neither basis
nor penalty, so it is fitted by backfitting:

```{code-cell} ipython3
from dataclasses import dataclass
from glmnet.gam import Smoother

@dataclass
class BinSmoother(Smoother):
    """A step function: the weighted mean of the residuals in each of `n_bins` bins."""
    n_bins: int = 10

    _label = 'bins'

    def setup(self, x):
        x = np.asarray(x, float)
        self.edges_ = np.quantile(x, np.linspace(0, 1, self.n_bins + 1))
        self.codes_ = self._codes(x)
        return self

    def _codes(self, x):
        return np.clip(np.searchsorted(self.edges_, x, side='right') - 1, 0, self.n_bins - 1)

    def _means(self, r, w):
        sw = np.bincount(self.codes_, w, self.n_bins)
        swr = np.bincount(self.codes_, w * r, self.n_bins)
        return np.divide(swr, sw, out=np.zeros(self.n_bins), where=sw > 0)

    def smooth(self, r):
        return self._means(r, self.w_)[self.codes_]

    def edf(self):
        # one mean per occupied bin, less the constant
        return float(np.sum(np.bincount(self.codes_, self.w_, self.n_bins) > 0) - 1)

    def finalize(self, f, r, coef=None):
        self.means_ = self._means(f, np.ones_like(f))
        return self

    def predict(self, x):
        return self.means_[self._codes(np.asarray(x, float))]

gam_bins = GAM(terms=[s('year', df=4), BinSmoother('age', n_bins=12), categorical('education')])
gam_bins.fit(Wage, Wage['wage'])
gam_bins.summary()
```

It is a term like the others: it is backfitted with the splines, and
plotted and predicted from (without bands: it has no basis):

```{code-cell} ipython3
ax = gam_bins.plot(terms=[1])[0]
ages = np.linspace(18, 80, 200)
ax.plot(ages, gam.partial_effect('s(age, df=5)', ages), label='s(age, df=5)')
ax.legend();
```

### A penalized smoother: a Fourier basis

A periodic smooth (of the hour of the day, the day of the year): $K$
sines and cosines, $f(x) = \sum_{k=1}^K a_k \sin(2\pi k x / P) + b_k
\cos(2\pi k x / P)$, penalized by the integral over a period of $f''^2$,
which is diagonal in the coefficients, $\frac P2 \sum_k (2\pi k/P)^4
(a_k^2 + b_k^2)$. The class gives the basis and the penalty; the fit, its
degrees of freedom, prediction, the joint solve, standard errors and REML
come with `PenalizedSmoother`. (The basis has no constant, which the
intercept takes.)

```{code-cell} ipython3
from glmnet.gam import PenalizedSmoother

@dataclass
class FourierSmoother(PenalizedSmoother):
    """A periodic smooth: K sines and cosines, penalized by the integral of f''^2 over a period."""
    K: int = 5
    period: float = 1.

    _label = 'fourier'

    def basis(self, x=None):
        # at the training rows (`self.x_`, kept by `setup`) or at `x`
        x = np.asarray(self.x_ if x is None else x, float)
        t = 2 * np.pi * np.outer(x, np.arange(1, self.K + 1)) / self.period
        return np.hstack([np.sin(t), np.cos(t)])

    def penalty_matrices(self):
        d = (2 * np.pi * np.arange(1, self.K + 1) / self.period) ** 4 * self.period / 2
        return [np.diag(np.r_[d, d])]
```

On the `Bikeshare` data, the bikers per hour, a Poisson GAM in the hour of
the day (a period of 24; `hr` is categorical, so as a number) and the
weather:

```{code-cell} ipython3
Bike = load_data('Bikeshare')
Bike['hour'] = Bike['hr'].astype(int)
gam_bike = GAM(terms=[FourierSmoother('hour', K=12, period=24, lam=1e-3),
                      categorical('weathersit')],
               family=sm.families.Poisson())
gam_bike.fit(Bike, Bike['bikers'])
gam_bike.summary()
```

```{code-cell} ipython3
hours = np.linspace(0, 24, 97)
curve, std = gam_bike.partial_effect(0, hours, return_std=True)
fig, ax = plt.subplots()
ax.plot(hours, curve)
ax.fill_between(hours, curve - 2 * std, curve + 2 * std, alpha=0.3)
ax.set_xlabel('hour'); ax.set_ylabel('log bikers (centered)');
```

The curve is periodic (its value at 24 is its value at 0), with its two
rush hours, and has standard errors, from the basis and the penalty. With
`lam` left out, a GLMM estimates it by REML, as for the built-in smooths;
with almost 9,000 hours of counts, REML keeps most of the 24 degrees of
freedom:

```{code-cell} ipython3
import warnings
from glmnet.gam import GLMM, ExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore', ExperimentalWarning)
    glmm_bike = GLMM(terms=[FourierSmoother('hour', K=12, period=24), categorical('weathersit')],
                     family=sm.families.Poisson()).fit(Bike, Bike['bikers'])
glmm_bike.smoothing_.lam, glmm_bike.edf_
```

### From scikit-learn: a P-spline

A scikit-learn transformer's features are a basis: `TransformerSmoother`
fits the transformer on the training rows, as `setup`, and transforms any
values, as `basis`. With `SplineTransformer`'s B-splines and a penalty on the
second differences of their coefficients it is a P-spline (Eilers and
Marx, 1996), mgcv's `bs="ps"`. B-splines sum to 1, so their span has the
constant, which the intercept takes; as mgcv does, the smoother absorbs
the constraint that its fit sum to 0 over the training rows
(`sum_to_zero=True`, its basis `B Z`, `glmnet.gam.sum_to_zero`), and so
has one coefficient fewer than features.

```{code-cell} ipython3
from sklearn.preprocessing import SplineTransformer
from glmnet.gam import TransformerSmoother

pspline = TransformerSmoother('age', SplineTransformer(n_knots=20, degree=3),
                              penalties='difference', lam=0.1, label='pspline(age)')
gam_ps = GAM(terms=[s('year', df=4), pspline, categorical('education')])
gam_ps.fit(Wage, Wage['wage'])
gam_ps.summary()
```

Its curve is close to the smoothing spline's, with its standard errors:

```{code-cell} ipython3
ax = gam_ps.plot(terms=[1])[0]
ax.plot(ages, gam.partial_effect('s(age, df=5)', ages), label='s(age, df=5)')
ax.legend();
```

As a `PenalizedSmoother`, a GLMM estimates its smoothing parameter:

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter('ignore', ExperimentalWarning)
    glmm_ps = GLMM(terms=[s('year'), TransformerSmoother('age', SplineTransformer(n_knots=20)),
                          categorical('education')]).fit(Wage, Wage['wage'])
glmm_ps.smoothing_.lam, glmm_ps.edf_
```

`penalties` is `'difference'` (of `order`, 2 by default), `'ridge'`, or a
list of matrices on the features' coefficients (several: a list `lam`);
any transformer of one or more columns will do (`PolynomialFeatures`, a
`Pipeline`).

### A regressor as a smoother

Backfitting needs only a weighted fit to the partial residuals, so any
scikit-learn regressor that takes `sample_weight` is a smoother
(`RegressorSmoother`; Hastie and Tibshirani, 1990, chapter 9): a tree, a
nearest-neighbours fit. It has no basis, so it is a GAM's term only,
without standard errors, and its degrees of freedom are Ye's (1998)
generalized degrees of freedom, `tr(dF/dr)` estimated by fits to noise,
unless given (`df=`). For a smoother that is not linear, backfitting need
not converge (check `converged_`).

```{code-cell} ipython3
from sklearn.tree import DecisionTreeRegressor
from glmnet.gam import RegressorSmoother

gam_tree = GAM(terms=[s('year', df=4), RegressorSmoother('age', DecisionTreeRegressor(max_depth=3)),
                      categorical('education')]).fit(Wage, Wage['wage'])
gam_tree.converged_, gam_tree.edf_
```

The tree's 8 leaves cost far more than 7 degrees of freedom: its splits
are chosen from the data (Ye's point).

```{code-cell} ipython3
ax = gam_tree.plot(terms=[1])[0]
ax.plot(ages, gam.partial_effect('s(age, df=5)', ages), label='s(age, df=5)')
ax.legend();
```

A `PenalizedSmoother` with several penalties (one per column, say) gives
them all in `penalty_matrices()` and takes a list `lam`. The source of
`glmnet.gam.terms`, `tensor` and `cyclic` has the built-in smoothers, which
also give the cheaper or more stable versions of these methods (a banded
solve for a spline, a diagonal one for a random effect), and the hooks of
modified backfitting (`linear_columns()`).

## References

- Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
  additive models. *The Annals of Statistics* 17(2), 453–510.
- Cleveland, W. S., Grosse, E. and Shyu, W. M. (1992). Local regression
  models. In J. M. Chambers and T. J. Hastie (eds.), *Statistical Models in
  S*, chapter 8. Wadsworth & Brooks/Cole.
- Eilers, P. H. C. and Marx, B. D. (1996). Flexible smoothing with
  B-splines and penalties. *Statistical Science* 11(2), 89–121.
- Hastie, T. and Tibshirani, R. (1986). Generalized additive models.
  *Statistical Science* 1(3), 297–318.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Hastie, T. and Tibshirani, R. (1993). Varying-coefficient models.
  *Journal of the Royal Statistical Society, Series B* 55(4), 757–796.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under
  nonstandard conditions. *Proceedings of the Fifth Berkeley Symposium on
  Mathematical Statistics and Probability* 1, 221–233.
- James, G., Witten, D., Hastie, T., Tibshirani, R. and Taylor, J. (2023).
  *An Introduction to Statistical Learning, with Applications in Python*.
  Springer.
- Marra, G. and Wood, S. N. (2012). Coverage properties of confidence
  intervals for generalized additive model components. *Scandinavian
  Journal of Statistics* 39(1), 53–74.
- Nychka, D. (1988). Bayesian confidence intervals for smoothing splines.
  *Journal of the American Statistical Association* 83(404), 1134–1143.
- Wahba, G. (1983). Bayesian "confidence intervals" for the cross-validated
  smoothing spline. *Journal of the Royal Statistical Society, Series B*
  45(1), 133–150.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix
  estimator and a direct test for heteroskedasticity. *Econometrica* 48(4),
  817–838.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. *Journal of the American Statistical
  Association* 111(516), 1548–1563.
- Ye, J. (1998). On measuring and correcting the effects of data mining and
  model selection. *Journal of the American Statistical Association*
  93(441), 120–131.
