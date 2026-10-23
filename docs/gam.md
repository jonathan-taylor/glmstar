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
effects, or a random effect. `glmnet.gam` fits GAMs in two ways that give
the same fit:

- **local scoring with backfitting** (Hastie and Tibshirani, 1986, 1990),
  as R's `gam::gam`: IRLS, each working problem solved by smoothing the
  partial residuals of each term in turn;
- **penalized regression** on the terms' bases (Wood, 2017), as R's
  `mgcv::gam`: each working problem one penalized least squares fit. This
  gives the Bayesian covariance of the fit, and the smoothing parameters
  can be estimated by REML or ML (Wood, 2011).

It is built on glmstar's IRLS, so the families are those of `GLM`
(statsmodels'), with weights and offsets, and the Cox model.

This page follows the GAM part of the ISLR lab on nonlinear modeling
(chapter 7), on its `Wage` data from the `ISLP` package.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data

from glmnet.gam import GAM, GLMM, GAMControl, s, lo, re, linear, categorical, tensor_spline, varying

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

## Smoothing parameters by REML

:::{warning}
REML and ML are **experimental** in glmstar 0.5: the fits, smoothing
parameters and variance components agree with mgcv, lme4 and nlme in the
tests, but the interface may change and the optimization is less tested than
the rest of `glmnet.gam`. `GLMM(method='REML')` warns
(`glmnet.gam.ExperimentalWarning`); check `smoothing_.converged`, and
compare with fixed smoothing parameters or with R.
:::

Rather than choosing each term's degrees of freedom, they can be estimated.
A spline's penalty is that of a Gaussian prior on its coefficients, and
`method='REML'` (or `'ML'`) chooses the smoothing parameters, with the
dispersion, to maximize the restricted (or full) marginal likelihood, with
the coefficients integrated out by Laplace's method, as mgcv (Wood, 2011).
`s()` without `df` or `lam` is then estimated. (This uses JAX, the extra
`pip install glmstar[gam]`.)

```{code-cell} ipython3
gam_reml = GLMM(terms=[s('year'), s('age'), categorical('education')], method='REML')
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

`tensor_spline('age', 'year')` is a smooth function of both columns, the
tensor product of a cubic regression spline in each (mgcv's
`te(age, year, bs="cr")`), with a penalty per column. Each column then has
its own smoothing parameter, and the term does not depend on the columns'
scales. In place of the two smooths, it models an interaction of age and
year:

```{code-cell} ipython3
gam_te = GLMM(terms=[tensor_spline('age', 'year', k=(6, 4)), categorical('education')], method='REML')
gam_te.fit(Wage, Wage['wage'])
gam_te.summary()
```

The smoothing parameters, one per margin:

```{code-cell} ipython3
dict(zip(gam_te.smoothing_.terms, gam_te.smoothing_.lam))
```

A two-column term is drawn as filled contours:

```{code-cell} ipython3
ax = gam_te.plot(terms=[0])[0]
ax.figure.tight_layout()
```

Its penalties' null space is spanned by $1$, age, year and their product:
however large its smoothing parameters, the term keeps a linear
age-by-year interaction.

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
gam_by = GLMM(terms=by_class, method='REML').fit(Wage, Wage['wage'])
gam_by.smoothing_.lam, gam_by.terms_[1].level_edf()
```

With `pooled_lam=True` the levels share one smoothing parameter, estimated
by REML as well (mgcv's `id`): the curves are taken to be equally smooth.

```{code-cell} ipython3
by_class[1] = varying(s('age'), by='jobclass', pooled_lam=True)
gam_pooled = GLMM(terms=by_class, method='REML').fit(Wage, Wage['wage'])
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
                    categorical('education')],
             method='REML').fit(Wage, Wage['wage'])
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

A random effect is a term like the smooths: `re(group)` gives each group
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
gam_re = GLMM(terms=[s('year'), s('age'), categorical('education'), re('maritl')], method='REML')
gam_re.fit(Wage, Wage['wage'])
gam_re.variance_components_
```

The fixed effects of a categorical are centered, the random effects are not,
so compare each level with the married workers:

```{code-cell} ipython3
gam_fe = GLMM(terms=[s('year'), s('age'), categorical('education'), categorical('maritl')],
             method='REML').fit(Wage, Wage['wage'])
random, fixed = gam_re.terms_[3].effects_, gam_fe.terms_[3].effects_
pd.DataFrame({'workers': Wage['maritl'].value_counts().sort_index(),
              'random effect': random - random['2. Married'],
              'fixed effect': fixed - fixed['2. Married']})
```

Each random effect is shrunk towards 0, the more the smaller its group:
the 19 widowed workers' effect moves the most, from $-14.0$ to $-10.8$
(relative to the married workers, whose own effect is shrunk little).

## A Cox model

With glmstar's `CoxFamily`, the GAM is a proportional hazards model with
smooth terms, fitted by maximizing the penalized partial likelihood (with
Efron's or Breslow's ties, start times and strata). The response is a data
frame with the times and the status. On the `BrainCancer` data, the hazard
of death as a smooth function of the gross tumor volume `gtv`, with the
Karnofsky index `ki` linear and an effect for the diagnosis, the smoothing
parameter by REML, as mgcv's `cox.ph` (Wood, Pya and Säfken, 2016):

```{code-cell} ipython3
from glmnet.cox import CoxFamily

Brain = load_data('BrainCancer').dropna()
cox = GLMM(terms=[s('gtv'), linear('ki'), categorical('diagnosis')],
          family=CoxFamily(event_id='time', status_id='status', tie_breaking='efron'),
          method='REML')
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
