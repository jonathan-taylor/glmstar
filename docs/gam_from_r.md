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

# GAMs and mixed models from R

This page translates models written with R's `gam`, `mgcv`, `lme4` and
`nlme` into `glmnet.gam`. Each translation is checked against R in the
tests (`tests/gam/test_gam_*_r.py` and others). The tables say what is
compared, and how closely it agrees.

A model in `glmnet.gam` is a list of terms, each naming the columns of a
data frame it uses, rather than a formula:

```r
mgcv::gam(wage ~ s(year) + s(age) + education, data=Wage, method="REML")
```

is

```{code-cell} ipython3
import warnings
import numpy as np
import pandas as pd
from ISLP import load_data

from glmnet.gam import GAM, GLMM, s, categorical, ExperimentalWarning

warnings.simplefilter('ignore', ExperimentalWarning)

Wage = load_data('Wage')
fit = GLMM(terms=[s('year'), s('age'), categorical('education')], method='REML')
fit.fit(Wage, Wage['wage'])
fit.summary()
```

The response is the second argument of `fit`. Prior weights and an offset
are columns named by `weight_id` and `offset_id`; the family is a
statsmodels family (`sm.families.Binomial()`) or glmstar's `CoxFamily`.

The term functions have descriptive names. Shorter ones are a matter of
import, e.g. `from glmnet.gam import tensor_spline as te`; the tables below
write `random_effect` as `re` (`from glmnet.gam import random_effect as re`).

## From `gam::gam`

R's `gam` (Hastie and Tibshirani's local scoring with backfitting) is a
`GAM`, its smoothing fixed (mgcv's estimated smoothing, below, is a
`GLMM`):

| R (`gam`) | `glmnet.gam` | Checked against R |
|---|---|---|
| `s(x, df=4)` | `s('x', df=4)` (`df=4` by default in a `GAM`) | fits to about 1e-4, R's precision of the df |
| `lo(x, span=0.5, degree=1)` | `lo('x', span=0.5, degree=1)` | against R's `loess` and `gam::lo` |
| `x` (linear) | `'x'`, or `linear('x')` | |
| a factor | `'f'` (a categorical column) or `contrast('f')`, or `categorical('f')` (formerly `factor('f')`, deprecated) | |
| `x:f`, `poly(x, 2)`, `ns(x, df=3)` | `('x', 'f')`, `poly('x', 2)`, `ns('x', df=3)` (modelmatrix's terms, as items of `terms`) | against mgcv's parametric coefficients |
| `family=binomial`, `weights=`, `offset(o)` | `family=sm.families.Binomial()`, `outcome=Outcome('y', weight='w', offset='o')` (modelmatrix) | Gaussian, binomial, Poisson, Gamma |
| `cbind(s, n - s) ~ ...`, `offset(log(e))` | `outcome=Binomial('s', trials='n')`, `Outcome('y', offset=log_e)` with `log_e` a modelmatrix feature | |
| `predict(m, newdata, type="terms")` | `fit.partial_effect(term, x)` | |
| `plot(m, se=TRUE)` | `fit.plot()` | |

`s()`'s knots are those of R's `gam::s` and `smooth.spline`: all the
distinct values when there are fewer than 50, otherwise about 200 of them,
evenly spaced among the sorted distinct values.

## From `mgcv`

`GLMM(method='REML')` or `'ML'` estimates the smoothing parameters as
`mgcv::gam(method=...)` (Wood, 2011). See [REML and ML for GAMs: the
design](gam_reml.md).

| R (`mgcv`) | `glmnet.gam` | Checked against R |
|---|---|---|
| `s(x, bs="cr", k=..., knots=...)` | `s('x')`, mgcv given our knots | fits to rounding at fixed `sp`; REML/ML fits to 1e-4 (5e-5 on Wage), edf to 1e-3 |
| `s(x, sp=...)` | `s('x', lam=...)`, `lam = sp / (sum(w) * S.scale)` | to rounding |
| `te(x, z, bs="cr")` | `tensor_spline('x', 'z')` (mgcv's `place.knots`) | REML/ML fits to 1e-3 with other terms, edf to 1e-3, ML criterion to 1e-7 |
| `s(x, bs="cc", k=k)` | `cyclic_spline('x', k=k)`, with `period=(a, b)` for the cycle's ends | REML/ML fits to 1e-8, criterion to 1e-9 |
| `s(x, bs="cr", k=k)` | `cubic_regression_spline('x', k=k)` | REML/ML criterion as mgcv's, fits to 1e-5 |
| `s(x, by=f)` (any `bs`) | `categorical('f'), varying(s('x'), by='f')` (any smooth; include the categorical, as mgcv's `y ~ f + s(x, by=f)`) | the ML criterion to 1e-7 (identical: each level's constraint is mgcv's); Gaussian REML fits to 1e-7 |
| `s(x, by=f, id=1)` | `varying(s('x'), by='f', pooled_lam=True)` | as above |
| `s(x, by=z)`, `z` numeric | `varying(s('x'), by='z', as_categorical=False)` | as above |
| `te(x, z, by=f)`, `s(h, bs="cc", by=f)` | `varying(tensor_spline('x', 'z'), by='f')`, `varying(cyclic_spline('h'), by='f')` | ML criterion to 1e-9 (te) |
| `s(x, z, bs="gp", m=...)` | `gaussian_process('x', 'z', m=...)` | at fixed `sp` to rounding; REML/ML to mgcv's convergence |
| `s(g, bs="re")` | `re('g')` | `gam.vcomp` to 1e-4 |
| `s(g, x, bs="re")` | `re('g', by='x')` | |
| `paraPen=list(X=list(S1, S2))` | `penalized_columns(cols, [S1, S2])` | fits to 1e-3 (mgcv stops sooner on a flat criterion) |
| `family=quasipoisson()`, `quasibinomial()` | `family=QuasiPoisson()`, `QuasiBinomial()` (`glmnet.gam`) | the dispersion (REML/ML) to 1e-4, ML criterion to 1e-7 |
| `family=nb()` | `family=NegativeBinomial()` (`glmnet.gam`), `theta` in `fit.negative_binomial_theta_` | theta to 1e-4, ML criterion to 1e-7 |
| `family=negbin(theta)` | `family=NegativeBinomial(theta=theta)` | ML criterion to 1e-7 |
| `family=cox.ph(), weights=status` | `family=CoxFamily(), outcome=Surv('time', 'status')` (modelmatrix) | fixed `sp` and REML |
| `m$sp` | `fit.smoothing_.lam_raw`, then times each `S.scale` | |
| `m$gcv.ubre` | `fit.reml_score_` | ML: equal; REML: up to a constant |
| `m$Vp`, `m$Ve` | `fit.posterior_covariance_`, `fit.frequentist_covariance_` | |
| `m$edf` (per term) | `fit.edf_` | |
| `predict(m, se.fit=TRUE)` | `fit.predict(X, return_std=True)` | |
| `gam.vcomp(m)` | `fit.variance_components_` | |

Two differences in parametrization:

- **The smoothing parameters' scale.** mgcv rescales each penalty matrix
  (its `S.scale`) and weights the deviance by the prior weights. `lam` is
  on the scale of `GAM`'s objective, the deviance over twice the sum of
  the weights. So `sp = lam * sum(w) * S.scale`, and `smoothing_.lam_raw`
  is `lam * sum(w)`.
- **The REML criterion's constant.** It depends on the parametrization of
  the unpenalized coefficients. Ours are sum-to-zero constraints; mgcv's
  and lme4's are treatment contrasts for a factor. So REML scores agree up
  to a constant when there are factors among the fixed effects. ML scores
  agree directly.

Where the criterion is flat, typically along a smoothing parameter going
to infinity (a smooth that is nearly linear), mgcv stops sooner. There the
fits agree to about 1e-4 and our criterion is as low.

## From `lme4` and `nlme`

A random effect is a term whose penalty is the Gaussian prior on its
effects (see [Mixed models as GAMs](gam_mixed_models.md)). For a Gaussian
response the REML and ML criteria are exactly lme4's.

| R (`lme4`, `nlme`) | `glmnet.gam` | Checked against R |
|---|---|---|
| `(1 \| g)` | `re('g')` | `lmer`: standard deviations to 1e-4, fits to 1e-6, log-likelihood to 1e-8 |
| `(1 \| a) + (1 \| b)` (crossed) | `re('a'), re('b')` | `lmer` |
| `(1 \| a/b)` (nested) | `re('a'), re(['a', 'b'])` | `lmer` (to 1e-5), `nlme::lme` (to 1e-4) |
| `(x \|\| g)` | `re('g'), re('g', by='x')` | `lmer` |
| `(1 + x \| g)` | `correlated_random_effect('g', slopes=['x'])` | `lmer`: standard deviations and correlation to 1e-5, log-likelihood to 1e-8 |
| `pdCompSymm(~ cond - 1)` (nlme) | `correlated_random_effect('g', slopes=[dummies], intercept=False, covariance='compound_symmetry')` | `lme`: covariance to 1e-4, log-likelihood to 1e-7 |
| `pdDiag(~ x)` (nlme) | `correlated_random_effect('g', slopes=['x'], covariance='diagonal')` | equals `re('g'), re('g', by='x')` exactly |
| `REML=TRUE` / `FALSE` | `method='REML'` / `'ML'` | |
| `glmer(..., nAGQ=0)` | `family=...`, `method='ML'` | variances and linear predictor to 1e-4, log-likelihood to 1e-6 |
| `glmer(..., nAGQ=1)` | | to about 3e-2: glmer also maximizes the approximation in the fixed effects |
| `(1 \| obs)` (observation-level) | `re('obs')`, a column with a label per row | `glmer` |
| `VarCorr(m)` | `fit.variance_components_`, and a correlated term's `covariance_` | |
| `ranef(m)` | the term's `effects_` (`fit.terms_[j].effects_`) | |
| `fixef(m)` | `fit.fixed_effects_` (with standard errors): the intercept and the linear terms' `coef_` | |
| `fitted(m)` | `fit.predict(X)` | |

Two differences in what is reported:

- **Standard errors of the random effects.** `fit.plot()` and
  `partial_effect(..., return_std=True)` use the posterior covariance
  (mgcv's `Vp`), which includes the uncertainty in the fixed effects.
  lme4's conditional standard deviations (`ranef(m, condVar=TRUE)`) hold
  the fixed effects at their estimates, and are smaller.
- **Groups not seen in fitting** predict 0 for the term, as lme4's
  `allow.new.levels=TRUE`.

## Not available

These have no translation yet:

- shrinkage bases (`bs="ts"`, `select=TRUE`) and thin-plate splines
  (`bs="tp"`; `gaussian_process` and `tensor_spline` are the multi-column
  smooths);
- mgcv's `Vc` correction of the covariance for the uncertainty in the
  smoothing parameters;
- `bam()` and its discretized fits for very large data;
- other structured covariances of correlated random effects (beyond
  `pdSymm`, `pdCompSymm` and `pdDiag`), AR(1) residuals, variance
  functions;
- `glmer` with adaptive Gauss–Hermite quadrature (`nAGQ > 1`);
- Kenward–Roger or Satterthwaite degrees of freedom.

A P-spline (`bs="ps"`) is a `TransformerSmoother` of scikit-learn's
`SplineTransformer` ([Generalized additive
models](gam.md#from-scikit-learn-a-p-spline)), not compared with mgcv's.

## References

- Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015). Fitting linear
  mixed-effects models using lme4. *Journal of Statistical Software*
  67(1), 1–48.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Pinheiro, J. C. and Bates, D. M. (2000). *Mixed-Effects Models in S and
  S-PLUS*. Springer.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
