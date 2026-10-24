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

# Generalized linear mixed models

A `GLMM` is a GAM whose smoothing parameters are estimated. It has the
same terms as a `GAM` ([Generalized additive models](gam.md)): each
penalized term is a random effect, its penalty $\frac{\lambda}{2} c^\top
S c$ the Gaussian prior $c \sim N(0, \phi (\lambda S)^{-1})$ of its
coefficients ($\phi$ the dispersion), so a smoothing spline is a random
effect on its basis' wiggly part, as in mgcv, and `re(group)` is a random
intercept, as in lme4. A `GLMM` estimates the variances, the
smoothing parameters $\lambda$, by

- **REML** (`method='REML'`, the default) or **ML** (`method='ML'`): the
  restricted or full marginal likelihood, the coefficients integrated out
  by Laplace's method, as mgcv's `gam(method="REML")` (Wood, 2011) and
  lme4's `lmer`/`glmer`;
- or a method of one's own, a `glmnet.gam.SmoothingMethod` (the method of
  moments for crossed random effects in [Crossed random effects at
  scale](gam_crossed.md)).

Everything else, the fit at the estimated parameters, its standard
errors, plots and predictions, is the GAM's. A GAM's smoothing, given by
each term's `lam` or `df`, is fixed: `GAM(method='REML')` and
`GLMM(method='fixed')` still work, with a FutureWarning.

:::{warning}
REML and ML are **experimental** in glmstar 0.5: the fits, smoothing
parameters and variance components agree with mgcv, lme4 and nlme in the
tests, but the interface may change and the optimization is less tested than
the rest of `glmnet.gam`. `GLMM(method='REML')` warns
(`glmnet.gam.ExperimentalWarning`); check `smoothing_.converged`, and
compare with fixed smoothing parameters or with R.
:::

This page continues [the GAM page](gam.md) on the `Wage` data. The
estimates use JAX (`pip install glmstar[gam]`), except for the Cox model.

```{code-cell} ipython3
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data

from glmnet.gam import (GLMM, GAM, GAMControl, ExperimentalWarning, s, random_effect as re, linear, categorical,
                        tensor_spline, varying)

warnings.simplefilter('ignore', ExperimentalWarning)
Wage = load_data('Wage')
```

## Smoothing parameters by REML

A spline's penalty is that of a Gaussian prior on its coefficients, and
`GLMM(method='REML')` (the default, or `'ML'`) chooses the smoothing parameters, with the
dispersion, to maximize the restricted (or full) marginal likelihood, with
the coefficients integrated out by Laplace's method, as mgcv (Wood, 2011).
`s()` without `df` or `lam` is estimated.

```{code-cell} ipython3
gam_reml = GLMM(terms=[s('year'), s('age'), categorical('education')])
gam_reml.fit(Wage, Wage['wage'])
gam_reml.summary()
```

REML gives `year` little more than a line, and `age` about 5 degrees of
freedom; `edf_` is now mgcv's. The smoothing parameters, and the criterion
at its minimum:

```{code-cell} ipython3
gam_reml.smoothing_.lam, gam_reml.reml_score_
```

```{code-cell} ipython3
axes = gam_reml.plot()
axes[0].figure.tight_layout()
```

## Smooths of several columns: tensor products

A tensor product smooth (`tensor_spline`, mgcv's `te`; see [the GAM
page](gam.md)) has a penalty, and so a smoothing parameter, per column,
both estimated:

```{code-cell} ipython3
gam_te = GLMM(terms=[tensor_spline('age', 'year', k=(6, 4)), categorical('education')])
gam_te.fit(Wage, Wage['wage'])
gam_te.summary()
```

The smoothing parameters, one per margin:

```{code-cell} ipython3
dict(zip(gam_te.smoothing_.terms, gam_te.smoothing_.lam))
```

## Varying coefficients

`varying(term, by=...)` lets a smooth term vary with another column, the
varying-coefficient models of Hastie and Tibshirani (1993); it is mgcv's
`by=`. With a categorical `by`, it fits a curve for each level:
`varying(s('age'), by='jobclass')` is mgcv's `s(age, by=jobclass)`. Each
curve is zero off its level; as mgcv's, each is constrained like the term
alone (it sums to zero over the data's ages), so the model includes
`categorical('jobclass')` for the levels' means, as mgcv's
`wage ~ jobclass + s(age, by=jobclass)`. Each level has its own smoothing
parameter:

```{code-cell} ipython3
by_class = [categorical('jobclass'), varying(s('age'), by='jobclass'), s('year'),
            categorical('education')]
gam_by = GLMM(terms=by_class).fit(Wage, Wage['wage'])
gam_by.smoothing_.lam, gam_by.terms_[1].level_edf()
```

With `pooled_lam=True` the levels share one smoothing parameter, estimated
by REML as well (mgcv's `id`): the curves are taken to be equally smooth.

```{code-cell} ipython3
by_class[1] = varying(s('age'), by='jobclass', pooled_lam=True)
gam_pooled = GLMM(terms=by_class).fit(Wage, Wage['wage'])
gam_pooled.smoothing_.lam, gam_pooled.terms_[1].level_edf()
```

The term is drawn as a curve per level:

```{code-cell} ipython3
ax = gam_by.plot(terms=[1])[0]
ax.figure.tight_layout()
```

`by` is categorical, whatever its values (one coded 0 and 1 too). With
`as_categorical=False` it is a number and the term is `z f(x)`: the coefficient
of `z` is a smooth function of `x` (mgcv's `s(x, by=z)`, not centered). Here the yearly
change in wage, `year - 2006`'s coefficient, is a function of age, next to
the main curve of age. (A smooth of the year as well would duplicate its
linear trend, which the varying coefficient already has.)

```{code-cell} ipython3
Wage['year_c'] = Wage['year'] - 2006
gam_vc = GLMM(terms=[s('age'), varying(s('age'), by='year_c', as_categorical=False),
                     categorical('education')]).fit(Wage, Wage['wage'])
ax = gam_vc.plot(terms=[1])[0]
ax.figure.tight_layout()
```

Any smooth can vary: `varying(cyclic_spline('hour', period=(0, 24)),
by='workingday')`, `varying(tensor_spline('lon', 'lat'), by='year')`,
`varying(re('school'), by='sector')`. Those with a basis (all but `lo()`)
work with every method; `varying(lo('x'), by='f')` fits a loess per level,
by backfitting only. mgcv's `s(age, by=jobclass, bs="cr")` is
`varying(cubic_regression_spline('age'), by='jobclass')`.

## Random effects

A random effect is a term like the smooths: `random_effect(group)`
(imported as `re`) gives each group
its own intercept, penalized by its sum of squares, as the Gaussian prior
of a mixed model; `re(group, by=x)` its own slope in `x`. Its variance is
the dispersion over its smoothing parameter, and REML estimates it as the
REML of the mixed model (for the Gaussian, exactly). With a list of
columns, `re(['batch', 'cask'])` gives each combination its own effect, so
nested effects, lme4's `(1 | batch/cask)`, are `re('batch') + re(['batch',
'cask'])`.

Here marital status is a random effect, its levels' effects shrunk
towards 0, next to the fixed effect of education:

```{code-cell} ipython3
gam_re = GLMM(terms=[s('year'), s('age'), categorical('education'), re('maritl')])
gam_re.fit(Wage, Wage['wage'])
gam_re.variance_components_
```

The fixed effects of a categorical are centered, the random effects are not,
so compare each level with the married workers:

```{code-cell} ipython3
gam_fe = GLMM(terms=[s('year'), s('age'), categorical('education'),
                     categorical('maritl')]).fit(Wage, Wage['wage'])
random, fixed = gam_re.terms_[3].effects_, gam_fe.terms_[3].effects_
pd.DataFrame({'workers': Wage['maritl'].value_counts().sort_index(),
              'random effect': random - random['2. Married'],
              'fixed effect': fixed - fixed['2. Married']})
```

Each random effect is shrunk towards 0, the more the smaller its group:
the 19 widowed workers' effect moves the most, from $-14.0$ to $-10.8$
(relative to the married workers, whose own effect is shrunk little).

The standard errors of a GLMM are from the posterior covariance, which
already models the grouping through the random effects. A sandwich
covariance (`sandwich_covariance(groups=...)`, or a meat of your own from
a grouped bootstrap) is a check against a wrong variance model: see [Sandwich
(robust) standard errors](gam.md#sandwich-robust-standard-errors).

## Holding some parameters fixed

A term given `lam` (or `df`, matched at the prior weights and then held, as
mgcv's `sp`) keeps it, and the others are estimated. Here the spline in
year is given one, and REML estimates the rest:

```{code-cell} ipython3
some = GLMM(terms=[s('year', lam=1e3), s('age'), categorical('education')]).fit(Wage, Wage['wage'])
dict(zip(some.smoothing_.terms, some.smoothing_.lam))
```

A term with several penalties takes a list, with `None` for the
parameters to estimate: the tensor product's smoothing parameter in age
given, its one in year estimated,

```{code-cell} ipython3
half = GLMM(terms=[tensor_spline('age', 'year', k=(6, 4), lam=[1., None]),
                   categorical('education')]).fit(Wage, Wage['wage'])
dict(zip(half.smoothing_.terms, half.smoothing_.lam)), half.terms_[0].penalty_lams()
```

and the same for a curve per level, `varying(s('age'), by='jobclass',
lam=[2., None])`, or a `penalized_columns` block. (A correlated random
effect, `correlated_random_effect`, is given whole, by its `precision`.)
`smoothing_.terms` names the estimated parameters, and
`variance_components_` has them all, the fixed ones at their values.

## A Cox model

The Cox model of [the GAM page](gam.md), the hazard of death on the
`BrainCancer` data as a smooth function of the gross tumor volume `gtv`,
with the Karnofsky index `ki` linear and an effect for the diagnosis, the
smoothing parameter by REML, as mgcv's `cox.ph` (Wood, Pya and Säfken, 2016):

```{code-cell} ipython3
from glmnet.cox import CoxFamily

Brain = load_data('BrainCancer').dropna()
cox = GLMM(terms=[s('gtv'), linear('ki'), categorical('diagnosis')],
           family=CoxFamily(event_id='time', status_id='status', tie_breaking='efron'))
cox.fit(Brain, Brain[['time', 'status']])
cox.summary()
```

```{code-cell} ipython3
axes = cox.plot()
axes[0].figure.tight_layout()
```

REML finds no evidence of a nonlinear effect of the tumor volume: its
spline is a line, 1 degree of freedom (as mgcv's `cox.ph` gives). The
curves are on the scale of the log hazard; `predict` gives the linear
predictor (`'link'`) or the relative risk (`'response'`).

## Cross-validation

A `GLMM` is a scikit-learn estimator, as a GAM: `GridSearchCV` tunes
what REML does not estimate, such as the smoother of a column, the size of
a basis or a given smoothing parameter, with REML estimating the rest in
each fold. Its parameters leave out the ones it estimates (a term's `lam`
or `df` left as None). Here, the margins' sizes of the tensor product:

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV, KFold

search = GridSearchCV(GLMM(terms=[tensor_spline('age', 'year', k=(6, 4)),
                                  categorical('education')]),
                      {'terms__0__k': [(4, 3), (6, 4), (10, 5)]},
                      cv=KFold(5, shuffle=True, random_state=0)).fit(Wage, Wage['wage'])
pd.Series(search.cv_results_['mean_test_score'],
          index=[str(k) for k in search.cv_results_['param_terms__0__k']], name='CV score')
```

For a model with random effects the folds decide what is scored: with
`GroupKFold` by the grouping, the prediction of new groups (their effects
0); with `KFold`, of the groups seen.

## A family of one's own (JAX)

REML and ML need of a family only its inverse link and unit deviance $d(y,
\mu)$, differentiated in $\eta$; the fit needs its variance function,
which is the deviance's too, $V(\mu) = 2 / \partial^2_\mu d(\mu, \mu)$.
`glmnet.gam.JaxFamily` takes the two, written in JAX, and finds the rest
by `jax.grad`. The probit by hand, against statsmodels' probit:

```{code-cell} ipython3
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jax.scipy.stats import norm
import statsmodels.api as sm
from glmnet.gam import JaxFamily

def bernoulli_deviance(y, mu):
    return 2 * (xlogy(y, y) - xlogy(y, mu) + xlogy(1 - y, 1 - y) - xlogy(1 - y, 1 - mu))

probit = JaxFamily(linkinv=norm.cdf, unit_deviance=bernoulli_deviance, name='probit')

high = Wage[Wage['education'] != '1. < HS Grad'].copy()
high['high'] = (high['wage'] > 250).astype(float)
terms = [s('age'), re('education')]
by_hand = GLMM(terms, family=probit).fit(high, high['high'])
builtin = GLMM(terms, family=sm.families.Binomial(link=sm.families.links.Probit())).fit(high, high['high'])

def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.abs(b))

(mu, se), (mu0, se0) = by_hand.predict(high, return_std=True), builtin.predict(high, return_std=True)
pd.Series({'lam': rel(by_hand.smoothing_.lam, builtin.smoothing_.lam),
           'REML score': rel(by_hand.smoothing_.score, builtin.smoothing_.score),
           'edf': rel(by_hand.edf_, builtin.edf_),
           'fitted': rel(mu, mu0), 'std err': rel(se, se0)}, name='max relative difference')
```

They agree to the optimizer's precision, the REML scores to rounding.
The point is a family statsmodels or REML lacks. A cauchit, say, is one
line, `JaxFamily(lambda eta: 0.5 + jnp.arctan(eta) / np.pi,
bernoulli_deviance)`. A family with a free dispersion also gives its
saturated log-likelihood, `saturated_loglike(y, a, phi)` (summed over the
observations, `a` the prior weights), with `fixed_dispersion=False`. Any
family with a method `jax_pieces()` returning these four (as
`glmnet.gam.reml._jax_family` documents) has REML and ML, a statsmodels
family among them.

## Other methods, and large problems

`method` can be a `glmnet.gam.SmoothingMethod`: an object whose
`select(glmm, dispersion)` sets the estimated terms' parameters, using
the fit at given parameters (`GLMM.inner_fit()`), the terms' penalties and,
if it likes, the REML criterion's pieces (`glmnet.gam.reml`). [Crossed
random effects at scale](gam_crossed.md) writes two: the method of moments
of Gao and Owen for crossed random effects, in time linear in the number of
observations, and Schall's PQL for binary ones.

REML and ML factor the Hessian of the coefficients. It is dense, `P x P`,
up to `GAMControl(max_dense_coef=20000)` coefficients; above that, when
one random effect has all but at most that many of them, it is factored by
blocks, without the dense matrix (`GAMControl(reml_backend='blocked')`,
`glmnet.gam.blocked`): MixedModels.jl's blocked Cholesky for its first
grouping factor, the random effect's block diagonal and the other
coefficients' Schur complement dense ([the design](gam_reml.md)). [Mixed models](gam_mixed_models.md) compares the
fits with lme4's, and [REML and ML for GAMs: the design](gam_reml.md)
derives the criterion and its gradient.

## References

- Buja, A., Hastie, T. and Tibshirani, R. (1989). Linear smoothers and
  additive models. *The Annals of Statistics* 17(2), 453–510.
- Cleveland, W. S., Grosse, E. and Shyu, W. M. (1992). Local regression
  models. In J. M. Chambers and T. J. Hastie (eds.), *Statistical Models in
  S*, chapter 8. Wadsworth & Brooks/Cole.
- Hastie, T. and Tibshirani, R. (1986). Generalized additive models.
  *Statistical Science* 1(3), 297–318.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Hastie, T. and Tibshirani, R. (1993). Varying-coefficient models.
  *Journal of the Royal Statistical Society, Series B* 55(4), 757–796.
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
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. *Journal of the American Statistical
  Association* 111(516), 1548–1563.
